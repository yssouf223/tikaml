"""Prediction service API.

FastAPI server that loads trained models and serves predictions
for the tikaml-data-service.

GAGNE TEMPS:
- Automatic daily fixtures from API-Football
- Current standings and form
- H2H
- Pre-match 1X2 odds
- TikaML predictions
- Confidence/ranking layer
- TOP 5 daily predictions

IMPORTANT:
API_FOOTBALL_KEY and TIKA_API_KEY must remain server-side.
"""

import json
import logging
import os
import re
import secrets
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from scipy.stats import poisson

from src.lgbm_poisson import LGBMPoissonModel
from src.live_predictor import LivePredictor
from src import national_api
from src.inference import MatchPredictor


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("tika-server")

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = Path("models/corners")
YELLOW_MODEL_DIR = Path("models/yellows")
BACKFILL_DIR = Path("models/backfill_20260131")

MAX_GOALS = 7


# ═══════════════════════════════════════════════════════════════════
# API KEYS
# ═══════════════════════════════════════════════════════════════════

API_KEY = os.environ.get("TIKA_API_KEY", "")

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)

    log.warning(
        "No TIKA_API_KEY set, generated temporary API key."
    )


API_FOOTBALL_KEY = os.environ.get(
    "API_FOOTBALL_KEY",
    "",
)

API_FOOTBALL_URL = (
    "https://v3.football.api-sports.io"
)


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str = Security(api_key_header),
):
    """Validate the TikaML API key."""

    if (
        not key
        or not secrets.compare_digest(
            key,
            API_KEY,
        )
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ═══════════════════════════════════════════════════════════════════
# SUPPORTED LEAGUES
# ═══════════════════════════════════════════════════════════════════

SUPPORTED_LEAGUES = {

    39: {
        "name": "Premier League",
        "tika_code": "EPL",
    },

    140: {
        "name": "La Liga",
        "tika_code": "LL",
    },

    135: {
        "name": "Serie A",
        "tika_code": "SEA",
    },

    78: {
        "name": "Bundesliga",
        "tika_code": "BUN",
    },

    61: {
        "name": "Ligue 1",
        "tika_code": "LI1",
    },
}


# ═══════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════

FIXTURE_CACHE = {
    "date": None,
    "timestamp": 0,
    "fixtures": None,
}

FIXTURE_CACHE_SECONDS = 300


DATA_CACHE = {
    "standings": {},
    "h2h": {},
    "odds": {},
}


STANDINGS_CACHE_SECONDS = 3600
H2H_CACHE_SECONDS = 21600
ODDS_CACHE_SECONDS = 900


# ═══════════════════════════════════════════════════════════════════
# MODEL CONTAINERS
# ═══════════════════════════════════════════════════════════════════

class Models:

    goals: LGBMPoissonModel | None = None

    corners: LGBMPoissonModel | None = None

    yellows: LGBMPoissonModel | None = None

    version: str = "unknown"


models = Models()
backfill_models = Models()


# ═══════════════════════════════════════════════════════════════════
# GAGNE TEMPS PREDICTOR
# ═══════════════════════════════════════════════════════════════════

club_predictor: MatchPredictor | None = None


# ═══════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class GagneTempsRequest(BaseModel):

    home_team: str

    away_team: str

    league: str

    season: str

    match_date: str

    week: int | None = None


class MatchContext(BaseModel):

    minute: int | None = None

    second: int = 0

    period: str | None = None

    status: str | None = None

    home_score: int = 0

    away_score: int = 0

    home_team_id: str | None = None

    away_team_id: str | None = None

    home_red_cards: int = 0

    away_red_cards: int = 0

    home_corners: int = 0

    away_corners: int = 0

    home_yellows: int = 0

    away_yellows: int = 0


class PredictionRequest(BaseModel):

    match_id: int

    opta_match_id: str = ""

    prediction_type: str = "prematch"

    trigger: str = ""

    feature_vector: dict[str, float | int | None]

    match_context: MatchContext | None = None

    models: list[str] = Field(
        default_factory=lambda: [
            "goals",
            "corners",
            "yellows",
        ]
    )


class PredictionResponse(BaseModel):

    predictions: dict

    model_metadata: dict


# ═══════════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):

    """Load all models on startup."""

    global club_predictor

    log.info("Loading models...")

    t0 = time.time()

    # ───────────────────────────────────────────────────────────────
    # GOALS
    # ───────────────────────────────────────────────────────────────

    models.goals = LGBMPoissonModel.load(
        str(MODEL_DIR)
    )

    log.info(
        f"  Goals model loaded "
        f"({len(models.goals.feature_cols)} features)"
    )

    # ───────────────────────────────────────────────────────────────
    # CORNERS
    # ───────────────────────────────────────────────────────────────

    if CORNER_MODEL_DIR.exists():

        models.corners = LGBMPoissonModel.load(
            str(CORNER_MODEL_DIR)
        )

        log.info(
            f"  Corners model loaded "
            f"({len(models.corners.feature_cols)} features)"
        )

    # ───────────────────────────────────────────────────────────────
    # YELLOWS
    # ───────────────────────────────────────────────────────────────

    if YELLOW_MODEL_DIR.exists():

        models.yellows = LGBMPoissonModel.load(
            str(YELLOW_MODEL_DIR)
        )

        log.info(
            f"  Yellows model loaded "
            f"({len(models.yellows.feature_cols)} features)"
        )

    # ───────────────────────────────────────────────────────────────
    # VERSION
    # ───────────────────────────────────────────────────────────────

    meta_path = MODEL_DIR / "meta.json"

    if meta_path.exists():

        with open(meta_path) as f:

            meta = json.load(f)

        models.version = (
            f"lgbm-poisson-"
            f"{len(meta.get('feature_cols', []))}f"
        )

    # ───────────────────────────────────────────────────────────────
    # BACKFILL
    # ───────────────────────────────────────────────────────────────

    if BACKFILL_DIR.exists():

        bf_goals_dir = BACKFILL_DIR / "goals"
        bf_corners_dir = BACKFILL_DIR / "corners"
        bf_yellows_dir = BACKFILL_DIR / "yellows"

        if bf_goals_dir.exists():

            backfill_models.goals = (
                LGBMPoissonModel.load(
                    str(bf_goals_dir)
                )
            )

        if bf_corners_dir.exists():

            backfill_models.corners = (
                LGBMPoissonModel.load(
                    str(bf_corners_dir)
                )
            )

        if bf_yellows_dir.exists():

            backfill_models.yellows = (
                LGBMPoissonModel.load(
                    str(bf_yellows_dir)
                )
            )

        backfill_models.version = (
            "lgbm-poisson-backfill-20260131"
        )

        log.info(
            "  Backfill models loaded"
        )

    # ───────────────────────────────────────────────────────────────
    # NATIONAL MODEL
    # ───────────────────────────────────────────────────────────────

    try:

        nm = national_api.load_national()

        log.info(
            f"  National model loaded "
            f"({len(nm.attack)} teams)"
        )

    except Exception as e:

        log.warning(
            f"  National model not loaded: {e}"
        )

    # ───────────────────────────────────────────────────────────────
    # GAGNE TEMPS
    # ───────────────────────────────────────────────────────────────

    try:

        features_path = Path(
            "data/opta/processed/features.csv"
        )

        if not features_path.exists():

            raise FileNotFoundError(
                f"features.csv not found: "
                f"{features_path}"
            )

        log.info(
            f"  Loading GAGNE TEMPS data: "
            f"{features_path}"
        )

        club_predictor = MatchPredictor(
            features_path=str(
                features_path
            )
        )

        club_predictor.load_model()

        log.info(
            f"  GAGNE TEMPS predictor loaded "
            f"({len(club_predictor.df)} "
            f"historical matches)"
        )

    except Exception as e:

        club_predictor = None

        log.exception(
            f"  GAGNE TEMPS predictor "
            f"failed to load: {e}"
        )

    # ───────────────────────────────────────────────────────────────
    # API-FOOTBALL
    # ───────────────────────────────────────────────────────────────

    if API_FOOTBALL_KEY:

        log.info(
            "  API-Football integration enabled"
        )

    else:

        log.warning(
            "  API-Football integration disabled: "
            "API_FOOTBALL_KEY is not configured"
        )

    log.info(
        f"  All models loaded in "
        f"{time.time() - t0:.1f}s"
    )

    yield

    log.info(
        "Shutting down"
    )


# ═══════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="TikaML Prediction Service",
    version="2.0.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════
# NATIONAL ROUTES
# ═══════════════════════════════════════════════════════════════════

app.include_router(
    national_api.router,
    dependencies=[
        Depends(verify_api_key)
    ],
)


# ═══════════════════════════════════════════════════════════════════
# GENERIC FEATURE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _build_feature_df(
    feature_vector: dict,
    feature_list: list[str],
) -> pd.DataFrame:

    row = {}

    for col in feature_list:

        val = feature_vector.get(
            col
        )

        row[col] = (
            float(val)
            if val is not None
            else np.nan
        )

    return pd.DataFrame(
        [row]
    )


def _goals_over_under(
    matrix: np.ndarray,
) -> dict:

    ou = {}

    for line in [
        1.5,
        2.5,
        3.5,
    ]:

        p_over = sum(
            matrix[i, j]

            for i in range(MAX_GOALS)

            for j in range(MAX_GOALS)

            if i + j > line
        )

        ou[str(line)] = {

            "over":
                round(
                    float(p_over),
                    4,
                ),

            "under":
                round(
                    float(1 - p_over),
                    4,
                ),
        }

    return ou


def _poisson_over_under(
    lambda_total: float,
    lines: list[float],
) -> dict:

    ou = {}

    for line in lines:

        p_over = float(
            1
            - poisson.cdf(
                int(line),
                lambda_total,
            )
        )

        ou[str(line)] = {

            "over":
                round(
                    p_over,
                    4,
                ),

            "under":
                round(
                    1 - p_over,
                    4,
                ),
        }

    return ou


def _live_over_under(
    lambda_remaining: float,
    current_total: int,
    lines: list[float],
) -> dict:

    ou = {}

    for line in lines:

        needed = (
            line
            - current_total
        )

        if needed <= 0:

            ou[str(line)] = {
                "over": 1.0,
                "under": 0.0,
            }

        else:

            p_over = float(
                1
                - poisson.cdf(
                    int(needed),
                    lambda_remaining,
                )
            )

            ou[str(line)] = {

                "over":
                    round(
                        p_over,
                        4,
                    ),

                "under":
                    round(
                        1 - p_over,
                        4,
                    ),
            }

    return ou


# ═══════════════════════════════════════════════════════════════════
# PREMATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_prematch(
    feature_vector: dict,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    # ───────────────────────────────────────────────────────────────
    # GOALS
    # ───────────────────────────────────────────────────────────────

    if (
        "goals" in requested_models
        and m.goals
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.goals.feature_cols,
        )

        lh, la = (
            m.goals.predict_lambdas(
                feat_df
            )
        )

        lh = float(lh[0])
        la = float(la[0])

        matrix = (
            m.goals.predict_score_matrix(
                lh,
                la,
                MAX_GOALS,
            )
        )

        p_home = float(
            np.tril(
                matrix,
                -1,
            ).sum()
        )

        p_draw = float(
            np.trace(matrix)
        )

        p_away = float(
            np.triu(
                matrix,
                1,
            ).sum()
        )

        total = (
            p_home
            + p_draw
            + p_away
        )

        if total > 0:

            p_home /= total
            p_draw /= total
            p_away /= total

        score_matrix = [

            [
                round(
                    float(
                        matrix[i, j]
                    ),
                    4,
                )

                for j in range(
                    MAX_GOALS
                )
            ]

            for i in range(
                MAX_GOALS
            )
        ]

        best_i, best_j = divmod(
            int(
                np.argmax(matrix)
            ),
            MAX_GOALS,
        )

        recommended_score = {

            "home_goals":
                best_i,

            "away_goals":
                best_j,

            "prob":
                round(
                    float(
                        matrix[
                            best_i,
                            best_j
                        ]
                    ),
                    4,
                ),

            "label":
                f"{best_i}-{best_j}",
        }

        predictions["goals"] = {

            "home_win":
                round(
                    p_home,
                    4,
                ),

            "draw":
                round(
                    p_draw,
                    4,
                ),

            "away_win":
                round(
                    p_away,
                    4,
                ),

            "expected_home":
                round(
                    lh,
                    4,
                ),

            "expected_away":
                round(
                    la,
                    4,
                ),

            "predicted_total":
                round(
                    lh + la,
                    4,
                ),

            "over_under":
                _goals_over_under(
                    matrix
                ),

            "score_matrix":
                score_matrix,

            "recommended_score":
                recommended_score,
        }

    # ───────────────────────────────────────────────────────────────
    # CORNERS
    # ───────────────────────────────────────────────────────────────

    if (
        "corners" in requested_models
        and m.corners
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.corners.feature_cols,
        )

        clh, cla = (
            m.corners.predict_lambdas(
                feat_df
            )
        )

        clh = float(clh[0])
        cla = float(cla[0])

        predictions["corners"] = {

            "expected_home":
                round(
                    clh,
                    4,
                ),

            "expected_away":
                round(
                    cla,
                    4,
                ),

            "predicted_total":
                round(
                    clh + cla,
                    4,
                ),

            "over_under":
                _poisson_over_under(
                    clh + cla,
                    [
                        8.5,
                        9.5,
                        10.5,
                        11.5,
                    ],
                ),
        }

    # ───────────────────────────────────────────────────────────────
    # YELLOWS
    # ───────────────────────────────────────────────────────────────

    if (
        "yellows" in requested_models
        and m.yellows
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.yellows.feature_cols,
        )

        ylh, yla = (
            m.yellows.predict_lambdas(
                feat_df
            )
        )

        ylh = float(ylh[0])
        yla = float(yla[0])

        predictions["yellows"] = {

            "expected_home":
                round(
                    ylh,
                    4,
                ),

            "expected_away":
                round(
                    yla,
                    4,
                ),

            "predicted_total":
                round(
                    ylh + yla,
                    4,
                ),

            "over_under":
                _poisson_over_under(
                    ylh + yla,
                    [
                        2.5,
                        3.5,
                        4.5,
                        5.5,
                    ],
                ),
        }

    return predictions


# ═══════════════════════════════════════════════════════════════════
# LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════

def predict_live(
    feature_vector: dict,
    ctx: MatchContext,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    minute = ctx.minute or 0

    # ───────────────────────────────────────────────────────────────
    # LIVE GOALS
    # ───────────────────────────────────────────────────────────────

    if (
        "goals" in requested_models
        and m.goals
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.goals.feature_cols,
        )

        lh, la = (
            m.goals.predict_lambdas(
                feat_df
            )
        )

        lh = float(lh[0])
        la = float(la[0])

        lp = LivePredictor(
            lh,
            la,
            rho=m.goals.rho,
            max_goals=MAX_GOALS,
        )

        lp.update(
            minute=minute,
            home_goals=ctx.home_score,
            away_goals=ctx.away_score,
            home_red_cards=ctx.home_red_cards,
            away_red_cards=ctx.away_red_cards,
        )

        live = lp.get_probabilities()

        rem = live["remaining_matrix"]

        matrix = np.zeros(
            (
                MAX_GOALS,
                MAX_GOALS,
            )
        )

        for i in range(MAX_GOALS):

            for j in range(MAX_GOALS):

                ri = (
                    i
                    - ctx.home_score
                )

                rj = (
                    j
                    - ctx.away_score
                )

                if (
                    0 <= ri < MAX_GOALS
                    and
                    0 <= rj < MAX_GOALS
                ):

                    matrix[i, j] = (
                        rem[ri, rj]
                    )

        if matrix.sum() > 0:

            matrix /= matrix.sum()

        score_matrix = [

            [
                round(
                    float(
                        matrix[i, j]
                    ),
                    4,
                )

                for j in range(
                    MAX_GOALS
                )
            ]

            for i in range(
                MAX_GOALS
            )
        ]

        best_i, best_j = divmod(
            int(
                np.argmax(matrix)
            ),
            MAX_GOALS,
        )

        recommended_score = {

            "home_goals":
                best_i,

            "away_goals":
                best_j,

            "prob":
                round(
                    float(
                        matrix[
                            best_i,
                            best_j
                        ]
                    ),
                    4,
                ),

            "label":
                f"{best_i}-{best_j}",
        }

        predictions["goals"] = {

            "home_win":
                round(
                    float(
                        live[
                            "probs_1x2"
                        ][0]
                    ),
                    4,
                ),

            "draw":
                round(
                    float(
                        live[
                            "probs_1x2"
                        ][1]
                    ),
                    4,
                ),

            "away_win":
                round(
                    float(
                        live[
                            "probs_1x2"
                        ][2]
                    ),
                    4,
                ),

            "expected_home":
                round(
                    lh,
                    4,
                ),

            "expected_away":
                round(
                    la,
                    4,
                ),

            "predicted_total":
                round(
                    lh + la,
                    4,
                ),

            "over_under": {

                str(k): {

                    "over":
                        round(
                            v["over"],
                            4,
                        ),

                    "under":
                        round(
                            v["under"],
                            4,
                        ),
                }

                for k, v in live[
                    "over_under"
                ].items()
            },

            "lambda_remaining_home":
                round(
                    float(
                        live[
                            "lambda_remaining"
                        ][0]
                    ),
                    4,
                ),

            "lambda_remaining_away":
                round(
                    float(
                        live[
                            "lambda_remaining"
                        ][1]
                    ),
                    4,
                ),

            "next_goal": {

                k:
                    round(
                        float(v),
                        4,
                    )

                for k, v in live[
                    "next_goal"
                ].items()
            },

            "score_matrix":
                score_matrix,

            "recommended_score":
                recommended_score,
        }

    # ───────────────────────────────────────────────────────────────
    # LIVE CORNERS / CARDS
    # ───────────────────────────────────────────────────────────────

    r = max(
        0,
        (90 - minute) / 90,
    )

    if (
        "corners" in requested_models
        and m.corners
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.corners.feature_cols,
        )

        clh, cla = (
            m.corners.predict_lambdas(
                feat_df
            )
        )

        clh = float(clh[0])
        cla = float(cla[0])

        predictions["corners"] = {

            "expected_home":
                round(
                    clh,
                    4,
                ),

            "expected_away":
                round(
                    cla,
                    4,
                ),

            "predicted_total":
                round(
                    clh + cla,
                    4,
                ),

            "lambda_remaining_home":
                round(
                    clh * r,
                    4,
                ),

            "lambda_remaining_away":
                round(
                    cla * r,
                    4,
                ),

            "over_under":
                _live_over_under(
                    (clh + cla) * r,
                    (
                        ctx.home_corners
                        + ctx.away_corners
                    ),
                    [
                        8.5,
                        9.5,
                        10.5,
                        11.5,
                    ],
                ),
        }

    if (
        "yellows" in requested_models
        and m.yellows
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.yellows.feature_cols,
        )

        ylh, yla = (
            m.yellows.predict_lambdas(
                feat_df
            )
        )

        ylh = float(ylh[0])
        yla = float(yla[0])

        predictions["yellows"] = {

            "expected_home":
                round(
                    ylh,
                    4,
                ),

            "expected_away":
                round(
                    yla,
                    4,
                ),

            "predicted_total":
                round(
                    ylh + yla,
                    4,
                ),

            "lambda_remaining_home":
                round(
                    ylh * r,
                    4,
                ),

            "lambda_remaining_away":
                round(
                    yla * r,
                    4,
                ),

            "over_under":
                _live_over_under(
                    (ylh + yla) * r,
                    (
                        ctx.home_yellows
                        + ctx.away_yellows
                    ),
                    [
                        2.5,
                        3.5,
                        4.5,
                        5.5,
                    ],
                ),
        }

    return predictions


# ═══════════════════════════════════════════════════════════════════
# TEAM NAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def _normalize_team_name(
    name: str,
) -> str:

    if not name:
        return ""

    value = unicodedata.normalize(
        "NFKD",
        str(name),
    )

    value = "".join(
        c
        for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower()

    value = value.replace(
        "&",
        "and",
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    ).strip()


TEAM_ALIASES = {

    "fc schalke 04":
        "Schalke 04",

    "schalke 04":
        "Schalke 04",

    "inter milan":
        "Inter",

    "internazionale":
        "Inter",

    "fc internazionale milano":
        "Inter",

    "psg":
        "Paris Saint-Germain",

    "paris saint germain":
        "Paris Saint-Germain",

    "paris saint-germain":
        "Paris Saint-Germain",

    "man united":
        "Manchester United",

    "manchester united":
        "Manchester United",

    "man city":
        "Manchester City",

    "manchester city":
        "Manchester City",

    "athletic bilbao":
        "Athletic Club",

    "athletic club":
        "Athletic Club",

    "club atletico de madrid":
        "Atletico Madrid",

    "atletico madrid":
        "Atletico Madrid",

    "real betis balompie":
        "Real Betis",

    "real betis":
        "Real Betis",

    "sporting de gijon":
        "Sporting Gijon",

    "sporting gijon":
        "Sporting Gijon",

    "paris fc":
        "Paris FC",
}


def _build_team_index():

    """
    Build normalized index from features.csv.
    """

    if club_predictor is None:
        return {}

    df = club_predictor.df

    teams = set()

    if "home_team" in df.columns:

        teams.update(
            df["home_team"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if "away_team" in df.columns:

        teams.update(
            df["away_team"]
            .dropna()
            .astype(str)
            .tolist()
        )

    index = {}

    for team in teams:

        normalized = (
            _normalize_team_name(
                team
            )
        )

        if normalized:

            index.setdefault(
                normalized,
                team,
            )

    return index


def _match_dataset_team(
    api_team_name: str,
    team_index: dict,
):

    """
    Find exact TikaML dataset team.
    """

    if not api_team_name:
        return None

    normalized = (
        _normalize_team_name(
            api_team_name
        )
    )

    # 1. Exact match
    if normalized in team_index:

        return team_index[
            normalized
        ]

    # 2. Explicit alias
    alias = TEAM_ALIASES.get(
        normalized
    )

    if alias:

        alias_norm = (
            _normalize_team_name(
                alias
            )
        )

        if alias_norm in team_index:

            return team_index[
                alias_norm
            ]

    # 3. Legacy aliases
    aliases = {

        "inter milan": [
            "Inter",
            "Internazionale",
            "Inter Milan",
        ],

        "internazionale": [
            "Inter",
            "Internazionale",
        ],

        "inter": [
            "Inter",
            "Internazionale",
        ],

        "psg": [
            "Paris Saint-Germain",
            "Paris Saint Germain",
        ],

        "paris saint germain": [
            "Paris Saint-Germain",
            "Paris Saint Germain",
        ],

        "paris saint germain fc": [
            "Paris Saint-Germain",
            "Paris Saint Germain",
        ],

        "fc schalke 04": [
            "Schalke 04",
            "FC Schalke 04",
        ],
    }

    if normalized in aliases:

        for candidate in aliases[
            normalized
        ]:

            candidate_norm = (
                _normalize_team_name(
                    candidate
                )
            )

            if candidate_norm in team_index:

                return team_index[
                    candidate_norm
                ]

    # 4. Conservative token matching
    tokens = set(
        normalized.split()
    )

    if tokens:

        candidates = []

        for (
            team_norm,
            team_name,
        ) in team_index.items():

            team_tokens = set(
                team_norm.split()
            )

            intersection = (
                tokens
                & team_tokens
            )

            if (
                len(intersection) >= 2
                and (
                    tokens <= team_tokens
                    or
                    team_tokens <= tokens
                )
            ):

                candidates.append(
                    (
                        len(intersection),
                        team_name,
                    )
                )

        if candidates:

            candidates.sort(
                reverse=True
            )

            return candidates[0][1]

    return None


# ═══════════════════════════════════════════════════════════════════
# API-FOOTBALL GENERIC REQUEST
# ═══════════════════════════════════════════════════════════════════

def _api_football_get(
    endpoint: str,
    params: dict | None = None,
):
    """
    Generic API-Football GET helper.
    """

    if not API_FOOTBALL_KEY:

        raise HTTPException(
            status_code=503,
            detail=(
                "API_FOOTBALL_KEY is not configured."
            ),
        )

    params = params or {}

    query = urlencode(
        {
            k: v
            for k, v in params.items()
            if v is not None
        }
    )

    url = (
        f"{API_FOOTBALL_URL}"
        f"{endpoint}"
    )

    if query:
        url += f"?{query}"

    request = Request(
        url,
        headers={
            "x-apisports-key":
                API_FOOTBALL_KEY,

            "Accept":
                "application/json",

            "User-Agent":
                "GAGNE-TEMPS/2.0",
        },
        method="GET",
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            raw = response.read()

        data = json.loads(
            raw.decode(
                "utf-8"
            )
        )

    except HTTPError as e:

        log.error(
            f"API-Football HTTP "
            f"{e.code} on {endpoint}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"API-Football HTTP "
                f"{e.code}"
            ),
        )

    except URLError as e:

        log.error(
            f"API-Football connection "
            f"error on {endpoint}: {e}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to connect to "
                "API-Football."
            ),
        )

    except Exception as e:

        log.exception(
            f"API-Football parsing error "
            f"on {endpoint}"
        )

        raise HTTPException(
            status_code=502,
            detail=(
                f"API-Football error: {e}"
            ),
        )

    if data.get("errors"):

        raise HTTPException(
            status_code=502,
            detail={
                "message":
                    "API-Football returned an error.",

                "errors":
                    data.get("errors"),
            },
        )

    return data.get(
        "response",
        [],
    )


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

def _fetch_api_football_fixtures(
    fixture_date: str,
):

    return _api_football_get(
        "/fixtures",
        {
            "date":
                fixture_date,
        },
    )


def _get_today_fixture_date():

    """
    Mali is UTC+0.
    """

    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d"
    )


def _get_cached_fixtures(
    fixture_date: str,
):

    now = time.time()

    if (
        FIXTURE_CACHE["date"]
        == fixture_date

        and
        FIXTURE_CACHE["fixtures"]
        is not None

        and
        (
            now
            - FIXTURE_CACHE["timestamp"]
        )
        < FIXTURE_CACHE_SECONDS
    ):

        return FIXTURE_CACHE[
            "fixtures"
        ]

    fixtures = (
        _fetch_api_football_fixtures(
            fixture_date
        )
    )

    FIXTURE_CACHE[
        "date"
    ] = fixture_date

    FIXTURE_CACHE[
        "timestamp"
    ] = now

    FIXTURE_CACHE[
        "fixtures"
    ] = fixtures

    return fixtures


# ═══════════════════════════════════════════════════════════════════
# CURRENT STANDINGS
# ═══════════════════════════════════════════════════════════════════

def _get_current_standings(
    league_id: int,
    season: int,
):

    cache_key = (
        f"{league_id}:{season}"
    )

    cached = DATA_CACHE[
        "standings"
    ].get(cache_key)

    now = time.time()

    if cached:

        if (
            now
            - cached["timestamp"]
        ) < STANDINGS_CACHE_SECONDS:

            return cached["data"]

    data = _api_football_get(
        "/standings",
        {
            "league":
                league_id,

            "season":
                season,
        },
    )

    DATA_CACHE[
        "standings"
    ][cache_key] = {

        "timestamp":
            now,

        "data":
            data,
    }

    return data


def _extract_team_standing(
    standings_response,
    team_id: int | None,
    team_name: str | None = None,
):

    if not standings_response:

        return None

    normalized_name = (
        _normalize_team_name(
            team_name or ""
        )
    )

    for block in standings_response:

        league = block.get(
            "league",
            {}
        )

        standings_groups = (
            league.get(
                "standings",
                []
            )
        )

        for group in standings_groups:

            for row in group:

                team = row.get(
                    "team",
                    {}
                )

                api_id = team.get(
                    "id"
                )

                api_name = team.get(
                    "name",
                    ""
                )

                if (
                    team_id is not None
                    and
                    api_id == team_id
                ):

                    return row

                if (
                    normalized_name
                    and
                    _normalize_team_name(
                        api_name
                    )
                    == normalized_name
                ):

                    return row

    return None


# ═══════════════════════════════════════════════════════════════════
# H2H
# ═══════════════════════════════════════════════════════════════════

def _get_h2h(
    home_id: int,
    away_id: int,
):

    cache_key = (
        f"{home_id}-{away_id}"
    )

    cached = DATA_CACHE[
        "h2h"
    ].get(cache_key)

    now = time.time()

    if cached:

        if (
            now
            - cached["timestamp"]
        ) < H2H_CACHE_SECONDS:

            return cached["data"]

    data = _api_football_get(
        "/fixtures/headtohead",
        {
            "h2h":
                f"{home_id}-{away_id}",

            "last":
                10,
        },
    )

    DATA_CACHE[
        "h2h"
    ][cache_key] = {

        "timestamp":
            now,

        "data":
            data,
    }

    return data


def _summarize_h2h(
    fixtures,
    home_id: int,
):

    if not fixtures:

        return {

            "matches":
                0,

            "home_wins":
                0,

            "draws":
                0,

            "away_wins":
                0,

            "home_goals":
                0,

            "away_goals":
                0,

            "home_win_pct":
                None,

            "draw_pct":
                None,

            "away_win_pct":
                None,
        }

    home_wins = 0
    draws = 0
    away_wins = 0

    home_goals = 0
    away_goals = 0

    for fixture in fixtures:

        teams = fixture.get(
            "teams",
            {}
        )

        goals = fixture.get(
            "goals",
            {}
        )

        h_team_id = (
            teams.get(
                "home",
                {}
            ).get(
                "id"
            )
        )

        h_goals = goals.get(
            "home"
        )

        a_goals = goals.get(
            "away"
        )

        if (
            h_goals is None
            or
            a_goals is None
        ):

            continue

        h_goals = int(
            h_goals
        )

        a_goals = int(
            a_goals
        )

        if h_team_id == home_id:

            home_goals += h_goals
            away_goals += a_goals

            if h_goals > a_goals:

                home_wins += 1

            elif h_goals == a_goals:

                draws += 1

            else:

                away_wins += 1

        else:

            home_goals += a_goals
            away_goals += h_goals

            if a_goals > h_goals:

                home_wins += 1

            elif a_goals == h_goals:

                draws += 1

            else:

                away_wins += 1

    total = (
        home_wins
        + draws
        + away_wins
    )

    return {

        "matches":
            total,

        "home_wins":
            home_wins,

        "draws":
            draws,

        "away_wins":
            away_wins,

        "home_goals":
            home_goals,

        "away_goals":
            away_goals,

        "home_win_pct":
            (
                round(
                    home_wins / total,
                    4,
                )
                if total
                else None
            ),

        "draw_pct":
            (
                round(
                    draws / total,
                    4,
                )
                if total
                else None
            ),

        "away_win_pct":
            (
                round(
                    away_wins / total,
                    4,
                )
                if total
                else None
            ),
    }


# ═══════════════════════════════════════════════════════════════════
# ODDS
# ═══════════════════════════════════════════════════════════════════

def _get_fixture_odds(
    fixture_id: int,
):

    cache_key = str(
        fixture_id
    )

    cached = DATA_CACHE[
        "odds"
    ].get(cache_key)

    now = time.time()

    if cached:

        if (
            now
            - cached["timestamp"]
        ) < ODDS_CACHE_SECONDS:

            return cached["data"]

    data = _api_football_get(
        "/odds",
        {
            "fixture":
                fixture_id,
        },
    )

    DATA_CACHE[
        "odds"
    ][cache_key] = {

        "timestamp":
            now,

        "data":
            data,
    }

    return data


def _extract_1x2_odds(
    odds_response,
):

    if not odds_response:

        return None

    for bookmaker_block in odds_response:

        bookmakers = (
            bookmaker_block.get(
                "bookmakers",
                []
            )
        )

        for bookmaker in bookmakers:

            bets = bookmaker.get(
                "bets",
                []
            )

            for bet in bets:

                bet_name = str(
                    bet.get(
                        "name",
                        ""
                    )
                ).lower()

                bet_id = bet.get(
                    "id"
                )

                if not (
                    bet_id == 1
                    or
                    "match winner"
                    in bet_name
                    or
                    bet_name == "winner"
                ):

                    continue

                values = bet.get(
                    "values",
                    []
                )

                home = None
                draw = None
                away = None

                for value in values:

                    label = str(
                        value.get(
                            "value",
                            ""
                        )
                    ).lower()

                    odd = value.get(
                        "odd"
                    )

                    try:

                        odd = float(
                            odd
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        continue

                    if label in (
                        "home",
                        "1",
                    ):

                        home = odd

                    elif label in (
                        "draw",
                        "x",
                    ):

                        draw = odd

                    elif label in (
                        "away",
                        "2",
                    ):

                        away = odd

                if (
                    home
                    and
                    draw
                    and
                    away
                ):

                    raw_h = 1 / home
                    raw_d = 1 / draw
                    raw_a = 1 / away

                    total = (
                        raw_h
                        + raw_d
                        + raw_a
                    )

                    return {

                        "home":
                            home,

                        "draw":
                            draw,

                        "away":
                            away,

                        "prob_home":
                            raw_h / total,

                        "prob_draw":
                            raw_d / total,

                        "prob_away":
                            raw_a / total,

                        "bookmaker":
                            bookmaker.get(
                                "name"
                            ),
                    }

    return None


# ═══════════════════════════════════════════════════════════════════
# CONFIDENCE ENGINE
# ═══════════════════════════════════════════════════════════════════

def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
):

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def _confidence_label(
    score: float,
):

    if score >= 78:

        return "TRÈS FORT"

    if score >= 68:

        return "FORT"

    if score >= 58:

        return "MOYEN"

    return "FAIBLE"


def _form_points(
    form: str,
):

    value = 0

    for char in (
        form or ""
    )[-5:]:

        if char == "W":

            value += 3

        elif char == "D":

            value += 1

    return value


def _calculate_confidence(
    prediction: dict,
    home_standing: dict | None,
    away_standing: dict | None,
    h2h_summary: dict | None,
    odds: dict | None,
):

    goals = prediction.get(
        "goals",
        {}
    )

    p_home = float(
        goals.get(
            "home_win",
            0
        )
    )

    p_draw = float(
        goals.get(
            "draw",
            0
        )
    )

    p_away = float(
        goals.get(
            "away_win",
            0
        )
    )

    probs = {

        "home":
            p_home,

        "draw":
            p_draw,

        "away":
            p_away,
    }

    predicted_side = max(
        probs,
        key=probs.get,
    )

    model_probability = (
        probs[predicted_side]
        * 100
    )

    # Base score.
    score = (
        45
        + max(
            0,
            model_probability - 33,
        ) * 0.65
    )

    evidence = []

    # ─────────────────────────────────────────────────────────────
    # CURRENT TABLE
    # ─────────────────────────────────────────────────────────────

    if (
        home_standing
        and away_standing
    ):

        hp = home_standing.get(
            "points"
        )

        ap = away_standing.get(
            "points"
        )

        if (
            hp is not None
            and
            ap is not None
        ):

            if (
                predicted_side == "home"
                and
                hp > ap
            ):

                score += 5

                evidence.append(
                    "Classement favorable domicile"
                )

            elif (
                predicted_side == "away"
                and
                ap > hp
            ):

                score += 5

                evidence.append(
                    "Classement favorable extérieur"
                )

    # ─────────────────────────────────────────────────────────────
    # FORM
    # ─────────────────────────────────────────────────────────────

    home_form = (
        home_standing.get(
            "form"
        )
        if home_standing
        else ""
    )

    away_form = (
        away_standing.get(
            "form"
        )
        if away_standing
        else ""
    )

    hf = _form_points(
        home_form
    )

    af = _form_points(
        away_form
    )

    if predicted_side == "home":

        if hf > af:

            score += 7

            evidence.append(
                "Forme récente favorable"
            )

    elif predicted_side == "away":

        if af > hf:

            score += 7

            evidence.append(
                "Forme récente favorable"
            )

    # ─────────────────────────────────────────────────────────────
    # H2H
    # ─────────────────────────────────────────────────────────────

    if h2h_summary:

        h2h_home = (
            h2h_summary.get(
                "home_win_pct"
            )
        )

        h2h_away = (
            h2h_summary.get(
                "away_win_pct"
            )
        )

        if predicted_side == "home":

            if (
                h2h_home is not None
                and
                h2h_home >= 0.50
            ):

                score += 5

                evidence.append(
                    "H2H favorable"
                )

        elif predicted_side == "away":

            if (
                h2h_away is not None
                and
                h2h_away >= 0.50
            ):

                score += 5

                evidence.append(
                    "H2H favorable"
                )

    # ─────────────────────────────────────────────────────────────
    # MARKET AGREEMENT
    # ─────────────────────────────────────────────────────────────

    market_probability = None

    if odds:

        market_probs = {

            "home":
                odds.get(
                    "prob_home"
                ),

            "draw":
                odds.get(
                    "prob_draw"
                ),

            "away":
                odds.get(
                    "prob_away"
                ),
        }

        market_probability = (
            market_probs.get(
                predicted_side
            )
        )

        if (
            market_probability
            is not None
        ):

            difference = abs(
                (
                    model_probability
                    / 100
                )
                - market_probability
            )

            if difference <= 0.05:

                score += 8

                evidence.append(
                    "Modèle et marché concordants"
                )

            elif difference <= 0.10:

                score += 3

            elif difference >= 0.18:

                score -= 8

                evidence.append(
                    "Désaccord important avec le marché"
                )

    score = _clamp(
        score
    )

    return {

        "score":
            round(
                score,
                2,
            ),

        "level":
            _confidence_label(
                score
            ),

        "predicted_side":
            predicted_side,

        "model_probability":
            round(
                model_probability,
                2,
            ),

        "market_probability":
            (
                round(
                    market_probability
                    * 100,
                    2,
                )
                if market_probability
                is not None
                else None
            ),

        "evidence":
            evidence[:5],
    }


# ═══════════════════════════════════════════════════════════════════
# BEST PICK
# ═══════════════════════════════════════════════════════════════════

def _build_best_pick(
    prediction: dict,
):

    goals = prediction.get(
        "goals",
        {}
    )

    if not goals:

        return None

    candidates = []

    home = float(
        goals.get(
            "home_win",
            0
        )
    )

    draw = float(
        goals.get(
            "draw",
            0
        )
    )

    away = float(
        goals.get(
            "away_win",
            0
        )
    )

    candidates.extend([

        {
            "market":
                "1",

            "probability":
                home,

            "label":
                "Victoire domicile",
        },

        {
            "market":
                "X",

            "probability":
                draw,

            "label":
                "Match nul",
        },

        {
            "market":
                "2",

            "probability":
                away,

            "label":
                "Victoire extérieur",
        },
    ])

    ou = goals.get(
        "over_under",
        {}
    )

    for line in (
        "1.5",
        "2.5",
        "3.5",
    ):

        if line not in ou:
            continue

        over = float(
            ou[line].get(
                "over",
                0
            )
        )

        under = float(
            ou[line].get(
                "under",
                0
            )
        )

        candidates.append({

            "market":
                f"Over {line}",

            "probability":
                over,

            "label":
                f"Plus de {line} buts",
        })

        candidates.append({

            "market":
                f"Under {line}",

            "probability":
                under,

            "label":
                f"Moins de {line} buts",
        })

    candidates.sort(
        key=lambda x:
            x["probability"],
        reverse=True,
    )

    best = candidates[0]

    return {

        "market":
            best["market"],

        "label":
            best["label"],

        "probability":
            round(
                best["probability"]
                * 100,
                2,
            ),
    }


# ═══════════════════════════════════════════════════════════════════
# JSON CLEANER
# ═══════════════════════════════════════════════════════════════════

def _clean_json(obj):

    if isinstance(
        obj,
        dict,
    ):

        return {
            str(k):
                _clean_json(v)

            for k, v in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):

        return [
            _clean_json(v)
            for v in obj
        ]

    if isinstance(
        obj,
        tuple,
    ):

        return [
            _clean_json(v)
            for v in obj
        ]

    if isinstance(
        obj,
        np.ndarray,
    ):

        return _clean_json(
            obj.tolist()
        )

    if isinstance(
        obj,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):

        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):

        if np.isnan(obj):

            return None

        return float(obj)

    if obj is None:

        return None

    try:

        if pd.isna(obj):

            return None

    except Exception:

        pass

    return obj


# ═══════════════════════════════════════════════════════════════════
# GAGNE TEMPS — SINGLE MATCH
# ═══════════════════════════════════════════════════════════════════

@app.post(
    "/gagne-temps/predict",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_predict(
    req: GagneTempsRequest,
):

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "GAGNE TEMPS predictor is not loaded. "
                "Check features.csv and model files."
            ),
        )

    try:

        match_date = pd.Timestamp(
            req.match_date
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid match_date. "
                "Use YYYY-MM-DD."
            ),
        )

    home_team = (
        req.home_team.strip()
    )

    away_team = (
        req.away_team.strip()
    )

    league = (
        req.league.strip().upper()
    )

    season = (
        req.season.strip()
    )

    if (
        not home_team
        or
        not away_team
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "home_team and away_team "
                "are required."
            ),
        )

    if (
        home_team.lower()
        == away_team.lower()
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "home_team and away_team "
                "must be different."
            ),
        )

    df = club_predictor.df

    known_home = (
        home_team
        in df["home_team"].values
        or
        home_team
        in df["away_team"].values
    )

    known_away = (
        away_team
        in df["home_team"].values
        or
        away_team
        in df["away_team"].values
    )

    if not known_home:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown home team: "
                f"{home_team}."
            ),
        )

    if not known_away:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown away team: "
                f"{away_team}."
            ),
        )

    t0 = time.time()

    result = club_predictor.predict(
        home_team=home_team,
        away_team=away_team,
        league=league,
        season=season,
        match_date=match_date,
        week=req.week,
    )

    elapsed = round(
        (
            time.time()
            - t0
        )
        * 1000,
        1,
    )

    result = _clean_json(
        result
    )

    return {

        "status":
            "success",

        "match": {

            "home_team":
                home_team,

            "away_team":
                away_team,

            "league":
                league,

            "season":
                season,

            "match_date":
                req.match_date,

            "week":
                req.week,
        },

        "prediction":
            result,

        "model_metadata": {

            "engine":
                "TikaML MatchPredictor",

            "version":
                models.version,

            "data_source":
                "data/opta/processed/features.csv",

            "elapsed_ms":
                elapsed,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# GAGNE TEMPS — AUTOMATIC TODAY V2
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/gagne-temps/today",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_today():

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "GAGNE TEMPS predictor "
                "is not loaded."
            ),
        )

    if not API_FOOTBALL_KEY:

        raise HTTPException(
            status_code=503,
            detail=(
                "API_FOOTBALL_KEY is not configured "
                "on Render."
            ),
        )

    fixture_date = (
        _get_today_fixture_date()
    )

    fixtures = (
        _get_cached_fixtures(
            fixture_date
        )
    )

    team_index = (
        _build_team_index()
    )

    results = []
    skipped = []

    for fixture in fixtures:

        try:

            league_data = fixture.get(
                "league",
                {}
            )

            league_id = league_data.get(
                "id"
            )

            if (
                league_id
                not in SUPPORTED_LEAGUES
            ):

                continue

            league_info = (
                SUPPORTED_LEAGUES[
                    league_id
                ]
            )

            fixture_info = fixture.get(
                "fixture",
                {}
            )

            fixture_id = fixture_info.get(
                "id"
            )

            teams = fixture.get(
                "teams",
                {}
            )

            home_obj = teams.get(
                "home",
                {}
            )

            away_obj = teams.get(
                "away",
                {}
            )

            home_api = home_obj.get(
                "name"
            )

            away_api = away_obj.get(
                "name"
            )

            home_api_id = home_obj.get(
                "id"
            )

            away_api_id = away_obj.get(
                "id"
            )

            # ─────────────────────────────────────────────────────
            # TEAM MATCHING
            # ─────────────────────────────────────────────────────

            home_team = (
                _match_dataset_team(
                    home_api,
                    team_index,
                )
            )

            away_team = (
                _match_dataset_team(
                    away_api,
                    team_index,
                )
            )

            if (
                not home_team
                or
                not away_team
            ):

                skipped.append({

                    "fixture_id":
                        fixture_id,

                    "home_team":
                        home_api,

                    "away_team":
                        away_api,

                    "league":
                        league_info[
                            "name"
                        ],

                    "reason":
                        "Team name not found "
                        "in TikaML dataset.",
                })

                continue

            # ─────────────────────────────────────────────────────
            # DATE
            # ─────────────────────────────────────────────────────

            date_time = fixture_info.get(
                "date"
            )

            match_date = (
                date_time[:10]
                if date_time
                else fixture_date
            )

            # ─────────────────────────────────────────────────────
            # WEEK
            # ─────────────────────────────────────────────────────

            round_name = league_data.get(
                "round"
            )

            week_number = None

            if round_name:

                round_match = re.search(
                    r"(\d+)",
                    str(round_name),
                )

                if round_match:

                    week_number = int(
                        round_match.group(1)
                    )

            # ─────────────────────────────────────────────────────
            # SEASON
            # ─────────────────────────────────────────────────────

            api_season = league_data.get(
                "season"
            )

            if api_season is None:

                api_season = datetime.now(
                    timezone.utc
                ).year

            # ─────────────────────────────────────────────────────
            # STANDINGS
            # ─────────────────────────────────────────────────────

            standings_response = (
                _get_current_standings(
                    league_id,
                    int(api_season),
                )
            )

            home_standing = (
                _extract_team_standing(
                    standings_response,
                    home_api_id,
                    home_api,
                )
            )

            away_standing = (
                _extract_team_standing(
                    standings_response,
                    away_api_id,
                    away_api,
                )
            )

            # ─────────────────────────────────────────────────────
            # H2H
            # ─────────────────────────────────────────────────────

            h2h_summary = None

            if (
                home_api_id
                and
                away_api_id
            ):

                h2h_response = (
                    _get_h2h(
                        home_api_id,
                        away_api_id,
                    )
                )

                h2h_summary = (
                    _summarize_h2h(
                        h2h_response,
                        home_api_id,
                    )
                )

            # ─────────────────────────────────────────────────────
            # ODDS
            # ─────────────────────────────────────────────────────

            odds = None

            if fixture_id:

                try:

                    odds_response = (
                        _get_fixture_odds(
                            fixture_id
                        )
                    )

                    odds = (
                        _extract_1x2_odds(
                            odds_response
                        )
                    )

                except Exception as odds_error:

                    log.warning(
                        f"Odds unavailable "
                        f"for fixture "
                        f"{fixture_id}: "
                        f"{odds_error}"
                    )

            # ─────────────────────────────────────────────────────
            # TIKAML
            # ─────────────────────────────────────────────────────

            t0 = time.time()

            prediction = (
                club_predictor.predict(
                    home_team=home_team,
                    away_team=away_team,
                    league=(
                        league_info[
                            "tika_code"
                        ]
                    ),
                    season=str(
                        api_season
                    ),
                    match_date=pd.Timestamp(
                        match_date
                    ),
                    week=week_number,
                    odds=odds,
                )
            )

            elapsed = round(
                (
                    time.time()
                    - t0
                ) * 1000,
                1,
            )

            prediction = _clean_json(
                prediction
            )

            # ─────────────────────────────────────────────────────
            # BEST PICK
            # ─────────────────────────────────────────────────────

            best_pick = (
                _build_best_pick(
                    prediction
                )
            )

            # ─────────────────────────────────────────────────────
            # CONFIDENCE
            # ─────────────────────────────────────────────────────

            confidence = (
                _calculate_confidence(
                    prediction,
                    home_standing,
                    away_standing,
                    h2h_summary,
                    odds,
                )
            )

            # ─────────────────────────────────────────────────────
            # CURRENT FORM
            # ─────────────────────────────────────────────────────

            home_form = (
                home_standing.get(
                    "form"
                )
                if home_standing
                else None
            )

            away_form = (
                away_standing.get(
                    "form"
                )
                if away_standing
                else None
            )

            # ─────────────────────────────────────────────────────
            # CURRENT TABLE
            # ─────────────────────────────────────────────────────

            table_context = {

                "home": {

                    "rank":
                        (
                            home_standing.get(
                                "rank"
                            )
                            if home_standing
                            else None
                        ),

                    "points":
                        (
                            home_standing.get(
                                "points"
                            )
                            if home_standing
                            else None
                        ),

                    "goals_diff":
                        (
                            home_standing.get(
                                "goalsDiff"
                            )
                            if home_standing
                            else None
                        ),

                    "form":
                        home_form,

                    "home_record":
                        (
                            home_standing.get(
                                "home"
                            )
                            if home_standing
                            else None
                        ),
                },

                "away": {

                    "rank":
                        (
                            away_standing.get(
                                "rank"
                            )
                            if away_standing
                            else None
                        ),

                    "points":
                        (
                            away_standing.get(
                                "points"
                            )
                            if away_standing
                            else None
                        ),

                    "goals_diff":
                        (
                            away_standing.get(
                                "goalsDiff"
                            )
                            if away_standing
                            else None
                        ),

                    "form":
                        away_form,

                    "away_record":
                        (
                            away_standing.get(
                                "away"
                            )
                            if away_standing
                            else None
                        ),
                },
            }

            results.append({

                "fixture_id":
                    fixture_id,

                "status":
                    fixture_info.get(
                        "status",
                        {},
                    ).get(
                        "short"
                    ),

                "kickoff":
                    date_time,

                "match": {

                    "home_team":
                        home_api,

                    "away_team":
                        away_api,

                    "tika_home_team":
                        home_team,

                    "tika_away_team":
                        away_team,
                },

                "league": {

                    "id":
                        league_id,

                    "name":
                        league_info[
                            "name"
                        ],

                    "tika_code":
                        league_info[
                            "tika_code"
                        ],
                },

                "season":
                    str(
                        api_season
                    ),

                "week":
                    week_number,

                "prediction":
                    prediction,

                "best_pick":
                    best_pick,

                "confidence":
                    confidence,

                "current_context": {

                    "standings":
                        table_context,

                    "h2h":
                        h2h_summary,

                    "odds":
                        odds,
                },

                "model_metadata": {

                    "engine":
                        "TikaML MatchPredictor",

                    "version":
                        models.version,

                    "source":
                        "API-Football + TikaML",

                    "live_context":
                        True,

                    "elapsed_ms":
                        elapsed,
                },
            })

        except Exception as e:

            log.exception(
                "Automatic GAGNE TEMPS "
                "prediction failed"
            )

            skipped.append({

                "fixture_id":
                    fixture.get(
                        "fixture",
                        {},
                    ).get(
                        "id"
                    ),

                "home_team":
                    fixture.get(
                        "teams",
                        {},
                    ).get(
                        "home",
                        {},
                    ).get(
                        "name"
                    ),

                "away_team":
                    fixture.get(
                        "teams",
                        {},
                    ).get(
                        "away",
                        {},
                    ).get(
                        "name"
                    ),

                "reason":
                    str(e),
            })

    # ═════════════════════════════════════════════════════════════
    # SORT
    # ═════════════════════════════════════════════════════════════

    results.sort(
        key=lambda item:
            item.get(
                "confidence",
                {}
            ).get(
                "score",
                0
            ),
        reverse=True,
    )

    # ═════════════════════════════════════════════════════════════
    # TOP 5
    # ═════════════════════════════════════════════════════════════

    top5 = []

    for rank, item in enumerate(
        results[:5],
        start=1,
    ):

        top5.append({

            "rank":
                rank,

            "fixture_id":
                item.get(
                    "fixture_id"
                ),

            "match":
                item.get(
                    "match"
                ),

            "league":
                item.get(
                    "league"
                ),

            "kickoff":
                item.get(
                    "kickoff"
                ),

            "best_pick":
                item.get(
                    "best_pick"
                ),

            "confidence":
                item.get(
                    "confidence"
                ),

            "recommended_score":
                item.get(
                    "prediction",
                    {}
                ).get(
                    "goals",
                    {}
                ).get(
                    "recommended_score"
                ),
        })

    return _clean_json({

        "status":
            "success",

        "date":
            fixture_date,

        "source":
            "API-Football + TikaML",

        "model":
            "TikaML MatchPredictor",

        "version":
            models.version,

        "supported_leagues":
            [
                league["name"]
                for league
                in SUPPORTED_LEAGUES.values()
            ],

        "total_api_fixtures":
            len(fixtures),

        "predictions_count":
            len(results),

        "skipped_count":
            len(skipped),

        "top_5":
            top5,

        "predictions":
            results,

        "skipped":
            skipped,

        "disclaimer":
            (
                "Predictions are statistical estimates "
                "and are not guarantees."
            ),
    })


# ═══════════════════════════════════════════════════════════════════
# GAGNE TEMPS — TOP 5
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/gagne-temps/top",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_top():

    """
    Lightweight endpoint for the mobile application.
    """

    data = await gagne_temps_today()

    return {

        "status":
            data["status"],

        "date":
            data["date"],

        "source":
            data["source"],

        "model":
            data["model"],

        "version":
            data["version"],

        "count":
            len(
                data["top_5"]
            ),

        "top_5":
            data["top_5"],

        "disclaimer":
            data["disclaimer"],
    }


# ═══════════════════════════════════════════════════════════════════
# STANDARD /predict
# ═══════════════════════════════════════════════════════════════════

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def predict(
    req: PredictionRequest,
):

    if models.goals is None:

        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )

    t0 = time.time()

    if (
        req.prediction_type
        == "live"

        and
        req.match_context
    ):

        predictions = predict_live(
            req.feature_vector,
            req.match_context,
            req.models,
        )

    else:

        predictions = predict_prematch(
            req.feature_vector,
            req.models,
        )

    elapsed = round(
        (
            time.time()
            - t0
        )
        * 1000,
        1,
    )

    log.info(
        f"predict "
        f"match_id={req.match_id} "
        f"type={req.prediction_type} "
        f"trigger={req.trigger} "
        f"models={list(predictions.keys())} "
        f"elapsed={elapsed}ms"
    )

    return PredictionResponse(

        predictions=predictions,

        model_metadata={

            "version":
                models.version,

            "prediction_type":
                req.prediction_type,

            "elapsed_ms":
                elapsed,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# BACKFILL
# ═══════════════════════════════════════════════════════════════════

@app.post(
    "/backfill",
    response_model=PredictionResponse,
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def backfill(
    req: PredictionRequest,
):

    if backfill_models.goals is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Backfill models not loaded"
            ),
        )

    t0 = time.time()

    predictions = predict_prematch(
        req.feature_vector,
        req.models,
        m=backfill_models,
    )

    elapsed = round(
        (
            time.time()
            - t0
        )
        * 1000,
        1,
    )

    log.info(
        f"backfill "
        f"match_id={req.match_id} "
        f"models={list(predictions.keys())} "
        f"elapsed={elapsed}ms"
    )

    return PredictionResponse(

        predictions=predictions,

        model_metadata={

            "version":
                backfill_models.version,

            "prediction_type":
                "prematch",

            "elapsed_ms":
                elapsed,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# MODEL STATUS
# ═══════════════════════════════════════════════════════════════════

@app.get(
    "/model-status"
)
async def model_status():

    return {

        "status":
            "ok",

        "models_loaded": {

            "goals":
                models.goals is not None,

            "corners":
                models.corners is not None,

            "yellows":
                models.yellows is not None,
        },

        "version":
            models.version,

        "backfill": {

            "loaded":
                backfill_models.goals
                is not None,

            "version":
                backfill_models.version,
        },

        "gagne_temps": {

            "loaded":
                club_predictor is not None,

            "features_loaded":
                (
                    club_predictor is not None
                    and
                    club_predictor.df is not None
                ),

            "historical_matches":
                (
                    len(
                        club_predictor.df
                    )
                    if (
                        club_predictor is not None
                        and
                        club_predictor.df is not None
                    )
                    else 0
                ),
        },

        "api_football": {

            "configured":
                bool(
                    API_FOOTBALL_KEY
                ),

            "integration":
                "API-Football + TikaML",

            "current_data": [

                "fixtures",
                "standings",
                "form",
                "home_away",
                "h2h",
                "odds",
            ],

            "supported_leagues":
                [
                    {

                        "id":
                            league_id,

                        "name":
                            league[
                                "name"
                            ],

                        "tika_code":
                            league[
                                "tika_code"
                            ],
                    }

                    for (
                        league_id,
                        league
                    )
                    in SUPPORTED_LEAGUES.items()
                ],
        },

        "national": {

            "loaded":
                (
                    national_api
                    .state
                    .model
                    is not None
                ),

            "teams":
                (
                    len(
                        national_api
                        .state
                        .model
                        .attack
                    )
                    if (
                        national_api
                        .state
                        .model
                    )
                    else 0
                ),
        },
    }
