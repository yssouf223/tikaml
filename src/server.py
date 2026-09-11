"""
TikaML Prediction Service
GAGNE TEMPS - OpenFootball Edition

Sources:
- OpenFootball football.json: fixtures + historical/current results
- TikaML: LightGBM + Poisson prediction engine

No API-Football dependency for GAGNE TEMPS.

OpenFootball:
https://github.com/openfootball/football.json

Important:
- TIKA_API_KEY remains server-side.
- No football external API key is required.
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


# ================================================================
# CONFIGURATION
# ================================================================

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


# ================================================================
# TIKA API KEY
# ================================================================

API_KEY = os.environ.get("TIKA_API_KEY", "")

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(
        "No TIKA_API_KEY configured. Temporary key generated."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str = Security(api_key_header),
):
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


# ================================================================
# SUPPORTED LEAGUES
# ================================================================

SUPPORTED_LEAGUES = {

    "EPL": {
        "name": "Premier League",
        "file": "en.1.json",
        "openfootball_code": "en.1",
    },

    "LL": {
        "name": "La Liga",
        "file": "es.1.json",
        "openfootball_code": "es.1",
    },

    "SEA": {
        "name": "Serie A",
        "file": "it.1.json",
        "openfootball_code": "it.1",
    },

    "BUN": {
        "name": "Bundesliga",
        "file": "de.1.json",
        "openfootball_code": "de.1",
    },

    "LI1": {
        "name": "Ligue 1",
        "file": "fr.1.json",
        "openfootball_code": "fr.1",
    },
}


OPENFOOTBALL_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)


# ================================================================
# OPENFOOTBALL CACHE
# ================================================================

OPENFOOTBALL_CACHE = {}

OPENFOOTBALL_CACHE_SECONDS = 1800


# ================================================================
# MODEL CONTAINERS
# ================================================================

class Models:

    goals: LGBMPoissonModel | None = None
    corners: LGBMPoissonModel | None = None
    yellows: LGBMPoissonModel | None = None
    version: str = "unknown"


models = Models()
backfill_models = Models()

club_predictor: MatchPredictor | None = None


# ================================================================
# REQUEST SCHEMAS
# ================================================================

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


# ================================================================
# LIFESPAN
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global club_predictor

    log.info("Loading models...")

    t0 = time.time()

    # GOALS
    models.goals = LGBMPoissonModel.load(
        str(MODEL_DIR)
    )

    log.info(
        f"Goals model loaded "
        f"({len(models.goals.feature_cols)} features)"
    )

    # CORNERS
    if CORNER_MODEL_DIR.exists():

        models.corners = LGBMPoissonModel.load(
            str(CORNER_MODEL_DIR)
        )

        log.info(
            f"Corners model loaded "
            f"({len(models.corners.feature_cols)} features)"
        )

    # YELLOWS
    if YELLOW_MODEL_DIR.exists():

        models.yellows = LGBMPoissonModel.load(
            str(YELLOW_MODEL_DIR)
        )

        log.info(
            f"Yellows model loaded "
            f"({len(models.yellows.feature_cols)} features)"
        )

    # VERSION
    meta_path = MODEL_DIR / "meta.json"

    if meta_path.exists():

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        models.version = (
            f"lgbm-poisson-"
            f"{len(meta.get('feature_cols', []))}f"
        )

    # BACKFILL
    if BACKFILL_DIR.exists():

        bf_goals = BACKFILL_DIR / "goals"
        bf_corners = BACKFILL_DIR / "corners"
        bf_yellows = BACKFILL_DIR / "yellows"

        if bf_goals.exists():
            backfill_models.goals = (
                LGBMPoissonModel.load(
                    str(bf_goals)
                )
            )

        if bf_corners.exists():
            backfill_models.corners = (
                LGBMPoissonModel.load(
                    str(bf_corners)
                )
            )

        if bf_yellows.exists():
            backfill_models.yellows = (
                LGBMPoissonModel.load(
                    str(bf_yellows)
                )
            )

        backfill_models.version = (
            "lgbm-poisson-backfill-20260131"
        )

        log.info("Backfill models loaded")

    # NATIONAL MODEL
    try:

        nm = national_api.load_national()

        log.info(
            f"National model loaded "
            f"({len(nm.attack)} teams)"
        )

    except Exception as e:

        log.warning(
            f"National model not loaded: {e}"
        )

    # GAGNE TEMPS
    try:

        features_path = Path(
            "data/opta/processed/features.csv"
        )

        if not features_path.exists():

            raise FileNotFoundError(
                f"features.csv not found: {features_path}"
            )

        club_predictor = MatchPredictor(
            features_path=str(
                features_path
            )
        )

        club_predictor.load_model()

        log.info(
            f"GAGNE TEMPS predictor loaded "
            f"({len(club_predictor.df)} historical matches)"
        )

    except Exception as e:

        club_predictor = None

        log.exception(
            f"GAGNE TEMPS predictor failed: {e}"
        )

    log.info(
        f"All models loaded in "
        f"{time.time() - t0:.1f}s"
    )

    yield

    log.info("Shutting down")


# ================================================================
# FASTAPI
# ================================================================

app = FastAPI(
    title="TikaML Prediction Service",
    version="3.0.0",
    lifespan=lifespan,
)


# ================================================================
# NATIONAL ROUTES
# ================================================================

app.include_router(
    national_api.router,
    dependencies=[
        Depends(verify_api_key)
    ],
)


# ================================================================
# GENERIC HELPERS
# ================================================================

def _build_feature_df(
    feature_vector: dict,
    feature_list: list[str],
) -> pd.DataFrame:

    row = {}

    for col in feature_list:

        val = feature_vector.get(col)

        row[col] = (
            float(val)
            if val is not None
            else np.nan
        )

    return pd.DataFrame([row])


def _goals_over_under(
    matrix: np.ndarray,
) -> dict:

    result = {}

    for line in [1.5, 2.5, 3.5]:

        p_over = sum(
            matrix[i, j]
            for i in range(MAX_GOALS)
            for j in range(MAX_GOALS)
            if i + j > line
        )

        result[str(line)] = {
            "over": round(float(p_over), 4),
            "under": round(
                float(1 - p_over),
                4,
            ),
        }

    return result


def _poisson_over_under(
    lambda_total: float,
    lines: list[float],
) -> dict:

    result = {}

    for line in lines:

        p_over = float(
            1
            - poisson.cdf(
                int(line),
                lambda_total,
            )
        )

        result[str(line)] = {
            "over": round(p_over, 4),
            "under": round(
                1 - p_over,
                4,
            ),
        }

    return result


def _live_over_under(
    lambda_remaining: float,
    current_total: int,
    lines: list[float],
) -> dict:

    result = {}

    for line in lines:

        needed = line - current_total

        if needed <= 0:

            result[str(line)] = {
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

            result[str(line)] = {
                "over": round(
                    p_over,
                    4,
                ),
                "under": round(
                    1 - p_over,
                    4,
                ),
            }

    return result


# ================================================================
# PREMATCH
# ================================================================

def predict_prematch(
    feature_vector: dict,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    # GOALS
    if "goals" in requested_models and m.goals:

        feat_df = _build_feature_df(
            feature_vector,
            m.goals.feature_cols,
        )

        lh, la = m.goals.predict_lambdas(
            feat_df
        )

        lh = float(lh[0])
        la = float(la[0])

        matrix = m.goals.predict_score_matrix(
            lh,
            la,
            MAX_GOALS,
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
                    float(matrix[i, j]),
                    4,
                )
                for j in range(MAX_GOALS)
            ]
            for i in range(MAX_GOALS)
        ]

        best_i, best_j = divmod(
            int(np.argmax(matrix)),
            MAX_GOALS,
        )

        predictions["goals"] = {

            "home_win":
                round(p_home, 4),

            "draw":
                round(p_draw, 4),

            "away_win":
                round(p_away, 4),

            "expected_home":
                round(lh, 4),

            "expected_away":
                round(la, 4),

            "predicted_total":
                round(lh + la, 4),

            "over_under":
                _goals_over_under(matrix),

            "score_matrix":
                score_matrix,

            "recommended_score": {

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
            },
        }

    # CORNERS
    if "corners" in requested_models and m.corners:

        feat_df = _build_feature_df(
            feature_vector,
            m.corners.feature_cols,
        )

        clh, cla = m.corners.predict_lambdas(
            feat_df
        )

        clh = float(clh[0])
        cla = float(cla[0])

        predictions["corners"] = {

            "expected_home":
                round(clh, 4),

            "expected_away":
                round(cla, 4),

            "predicted_total":
                round(clh + cla, 4),

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

    # YELLOWS
    if "yellows" in requested_models and m.yellows:

        feat_df = _build_feature_df(
            feature_vector,
            m.yellows.feature_cols,
        )

        ylh, yla = m.yellows.predict_lambdas(
            feat_df
        )

        ylh = float(ylh[0])
        yla = float(yla[0])

        predictions["yellows"] = {

            "expected_home":
                round(ylh, 4),

            "expected_away":
                round(yla, 4),

            "predicted_total":
                round(ylh + yla, 4),

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


# ================================================================
# LIVE
# ================================================================

def predict_live(
    feature_vector: dict,
    ctx: MatchContext,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    minute = ctx.minute or 0

    if "goals" in requested_models and m.goals:

        feat_df = _build_feature_df(
            feature_vector,
            m.goals.feature_cols,
        )

        lh, la = m.goals.predict_lambdas(
            feat_df
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
            (MAX_GOALS, MAX_GOALS)
        )

        for i in range(MAX_GOALS):

            for j in range(MAX_GOALS):

                ri = i - ctx.home_score
                rj = j - ctx.away_score

                if (
                    0 <= ri < MAX_GOALS
                    and
                    0 <= rj < MAX_GOALS
                ):

                    matrix[i, j] = rem[ri, rj]

        if matrix.sum() > 0:
            matrix /= matrix.sum()

        best_i, best_j = divmod(
            int(np.argmax(matrix)),
            MAX_GOALS,
        )

        predictions["goals"] = {

            "home_win":
                round(
                    float(
                        live["probs_1x2"][0]
                    ),
                    4,
                ),

            "draw":
                round(
                    float(
                        live["probs_1x2"][1]
                    ),
                    4,
                ),

            "away_win":
                round(
                    float(
                        live["probs_1x2"][2]
                    ),
                    4,
                ),

            "expected_home":
                round(lh, 4),

            "expected_away":
                round(la, 4),

            "predicted_total":
                round(lh + la, 4),

            "over_under": {
                str(k): {
                    "over": round(
                        v["over"],
                        4,
                    ),
                    "under": round(
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
                k: round(
                    float(v),
                    4,
                )
                for k, v in live[
                    "next_goal"
                ].items()
            },

            "recommended_score": {

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
            },
        }

    r = max(
        0,
        (90 - minute) / 90,
    )

    if "corners" in requested_models and m.corners:

        feat_df = _build_feature_df(
            feature_vector,
            m.corners.feature_cols,
        )

        clh, cla = m.corners.predict_lambdas(
            feat_df
        )

        clh = float(clh[0])
        cla = float(cla[0])

        predictions["corners"] = {

            "expected_home":
                round(clh, 4),

            "expected_away":
                round(cla, 4),

            "predicted_total":
                round(clh + cla, 4),

            "lambda_remaining_home":
                round(clh * r, 4),

            "lambda_remaining_away":
                round(cla * r, 4),

            "over_under":
                _live_over_under(
                    (clh + cla) * r,
                    ctx.home_corners
                    + ctx.away_corners,
                    [
                        8.5,
                        9.5,
                        10.5,
                        11.5,
                    ],
                ),
        }

    if "yellows" in requested_models and m.yellows:

        feat_df = _build_feature_df(
            feature_vector,
            m.yellows.feature_cols,
        )

        ylh, yla = m.yellows.predict_lambdas(
            feat_df
        )

        ylh = float(ylh[0])
        yla = float(yla[0])

        predictions["yellows"] = {

            "expected_home":
                round(ylh, 4),

            "expected_away":
                round(yla, 4),

            "predicted_total":
                round(ylh + yla, 4),

            "lambda_remaining_home":
                round(ylh * r, 4),

            "lambda_remaining_away":
                round(yla * r, 4),

            "over_under":
                _live_over_under(
                    (ylh + yla) * r,
                    ctx.home_yellows
                    + ctx.away_yellows,
                    [
                        2.5,
                        3.5,
                        4.5,
                        5.5,
                    ],
                ),
        }

    return predictions


# ================================================================
# TEAM NORMALIZATION
# ================================================================

def _normalize_team_name(name: str) -> str:

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
        [
            "Schalke 04",
            "FC Schalke 04",
        ],

    "inter milan":
        [
            "Inter",
            "Internazionale",
            "Inter Milan",
        ],

    "internazionale":
        [
            "Inter",
            "Internazionale",
        ],

    "psg":
        [
            "Paris Saint-Germain",
            "Paris Saint-Germain FC",
        ],

    "paris saint germain":
        [
            "Paris Saint-Germain",
            "Paris Saint-Germain FC",
        ],

    "paris saint germain fc":
        [
            "Paris Saint-Germain",
            "Paris Saint-Germain FC",
        ],

    "man united":
        [
            "Manchester United",
        ],

    "manchester united":
        [
            "Manchester United",
        ],

    "man city":
        [
            "Manchester City",
        ],

    "athletic bilbao":
        [
            "Athletic Club",
            "Athletic Bilbao",
        ],

    "atletico madrid":
        [
            "Atletico Madrid",
            "Club Atletico de Madrid",
        ],

    "real betis":
        [
            "Real Betis",
            "Real Betis Balompie",
        ],

    "sporting gijon":
        [
            "Sporting Gijon",
            "Sporting de Gijon",
        ],

    "rennes":
        [
            "Rennes",
            "Stade Rennais",
            "Stade Rennais FC 1901",
        ],

    "stade rennais":
        [
            "Rennes",
            "Stade Rennais",
            "Stade Rennais FC 1901",
        ],

    "marseille":
        [
            "Marseille",
            "Olympique de Marseille",
        ],

    "olympique de marseille":
        [
            "Marseille",
            "Olympique de Marseille",
        ],

    "sevilla":
        [
            "Sevilla",
            "Sevilla FC",
        ],

    "valencia":
        [
            "Valencia",
            "Valencia CF",
        ],

    "venezia":
        [
            "Venezia",
            "Venezia FC",
        ],

    "fiorentina":
        [
            "Fiorentina",
            "ACF Fiorentina",
        ],
}


def _build_team_index():

    if club_predictor is None:
        return {}

    df = club_predictor.df

    teams = set()

    for column in [
        "home_team",
        "away_team",
    ]:

        if column in df.columns:

            teams.update(
                df[column]
                .dropna()
                .astype(str)
                .tolist()
            )

    index = {}

    for team in teams:

        normalized = _normalize_team_name(
            team
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

    if not api_team_name:
        return None

    normalized = _normalize_team_name(
        api_team_name
    )

    # Exact
    if normalized in team_index:
        return team_index[normalized]

    # Alias
    for candidate in TEAM_ALIASES.get(
        normalized,
        [],
    ):

        candidate_norm = (
            _normalize_team_name(
                candidate
            )
        )

        if candidate_norm in team_index:

            return team_index[
                candidate_norm
            ]

    # Conservative matching
    tokens = set(
        normalized.split()
    )

    if not tokens:
        return None

    candidates = []

    for (
        team_norm,
        team_name,
    ) in team_index.items():

        team_tokens = set(
            team_norm.split()
        )

        intersection = (
            tokens & team_tokens
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
            key=lambda x: (
                x[0],
                len(x[1]),
            ),
            reverse=True,
        )

        return candidates[0][1]

    return None


# ================================================================
# OPENFOOTBALL HTTP
# ================================================================

def _download_openfootball(
    season_folder: str,
    filename: str,
):

    url = (
        f"{OPENFOOTBALL_BASE_URL}/"
        f"{season_folder}/"
        f"{filename}"
    )

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "GAGNE-TEMPS/3.0",
        },
        method="GET",
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            raw = response.read()

        return json.loads(
            raw.decode("utf-8")
        )

    except HTTPError as e:

        raise RuntimeError(
            f"OpenFootball HTTP {e.code}: {url}"
        )

    except URLError as e:

        raise RuntimeError(
            f"OpenFootball connection error: {e}"
        )

    except json.JSONDecodeError as e:

        raise RuntimeError(
            f"Invalid OpenFootball JSON: {e}"
        )


def _get_openfootball_dataset(
    tika_code: str,
    season: str,
):

    league = SUPPORTED_LEAGUES.get(
        tika_code
    )

    if not league:

        raise ValueError(
            f"Unsupported league: {tika_code}"
        )

    # TikaML usually uses 2026.
    # OpenFootball uses 2026-27.
    try:

        year = int(
            str(season)[:4]
        )

    except Exception:

        year = datetime.now(
            timezone.utc
        ).year

    season_folder = f"{year}-{str(year + 1)[-2:]}"

    cache_key = (
        f"{season_folder}:"
        f"{league['file']}"
    )

    cached = OPENFOOTBALL_CACHE.get(
        cache_key
    )

    now = time.time()

    if cached:

        if (
            now
            - cached["timestamp"]
        ) < OPENFOOTBALL_CACHE_SECONDS:

            return cached["data"]

    data = _download_openfootball(
        season_folder,
        league["file"],
    )

    OPENFOOTBALL_CACHE[
        cache_key
    ] = {

        "timestamp":
            now,

        "data":
            data,
    }

    log.info(
        f"OpenFootball loaded: "
        f"{season_folder}/{league['file']} "
        f"({len(data.get('matches', []))} matches)"
    )

    return data


# ================================================================
# OPENFOOTBALL MATCH HELPERS
# ================================================================

def _extract_score(match):

    score = match.get(
        "score"
    )

    if not score:
        return None

    if isinstance(
        score,
        list,
    ):

        if len(score) >= 2:
            return (
                int(score[0]),
                int(score[1]),
            )

        return None

    if isinstance(
        score,
        dict,
    ):

        ft = score.get(
            "ft"
        )

        if (
            isinstance(ft, list)
            and len(ft) >= 2
        ):

            try:

                return (
                    int(ft[0]),
                    int(ft[1]),
                )

            except (
                TypeError,
                ValueError,
            ):

                return None

    return None


def _match_is_finished(match):

    return (
        _extract_score(match)
        is not None
    )


def _openfootball_team_matches(
    matches,
    team_name,
    before_date=None,
):

    normalized = _normalize_team_name(
        team_name
    )

    result = []

    for match in matches:

        team1 = match.get(
            "team1",
            ""
        )

        team2 = match.get(
            "team2",
            ""
        )

        n1 = _normalize_team_name(
            team1
        )

        n2 = _normalize_team_name(
            team2
        )

        if (
            normalized != n1
            and
            normalized != n2
        ):

            continue

        date = match.get(
            "date"
        )

        if not date:
            continue

        if before_date and date >= before_date:
            continue

        if not _match_is_finished(match):
            continue

        result.append(match)

    result.sort(
        key=lambda x:
            x.get("date", ""),
        reverse=True,
    )

    return result


# ================================================================
# CURRENT STANDINGS FROM OPENFOOTBALL
# ================================================================

def _calculate_standings(
    matches,
    before_date=None,
):

    table = {}

    for match in matches:

        date = match.get(
            "date"
        )

        if not date:
            continue

        if before_date and date >= before_date:
            continue

        score = _extract_score(
            match
        )

        if score is None:
            continue

        team1 = match.get(
            "team1"
        )

        team2 = match.get(
            "team2"
        )

        if not team1 or not team2:
            continue

        g1, g2 = score

        for team in [
            team1,
            team2,
        ]:

            if team not in table:

                table[team] = {

                    "team":
                        team,

                    "played":
                        0,

                    "wins":
                        0,

                    "draws":
                        0,

                    "losses":
                        0,

                    "goals_for":
                        0,

                    "goals_against":
                        0,

                    "points":
                        0,

                    "home_played":
                        0,

                    "home_wins":
                        0,

                    "home_draws":
                        0,

                    "home_losses":
                        0,

                    "home_goals_for":
                        0,

                    "home_goals_against":
                        0,

                    "away_played":
                        0,

                    "away_wins":
                        0,

                    "away_draws":
                        0,

                    "away_losses":
                        0,

                    "away_goals_for":
                        0,

                    "away_goals_against":
                        0,
                }

        h = table[team1]
        a = table[team2]

        h["played"] += 1
        a["played"] += 1

        h["goals_for"] += g1
        h["goals_against"] += g2

        a["goals_for"] += g2
        a["goals_against"] += g1

        h["home_played"] += 1
        h["home_goals_for"] += g1
        h["home_goals_against"] += g2

        a["away_played"] += 1
        a["away_goals_for"] += g2
        a["away_goals_against"] += g1

        if g1 > g2:

            h["wins"] += 1
            h["points"] += 3

            a["losses"] += 1

            h["home_wins"] += 1
            a["away_losses"] += 1

        elif g1 < g2:

            a["wins"] += 1
            a["points"] += 3

            h["losses"] += 1

            a["away_wins"] += 1
            h["home_losses"] += 1

        else:

            h["draws"] += 1
            a["draws"] += 1

            h["points"] += 1
            a["points"] += 1

            h["home_draws"] += 1
            a["away_draws"] += 1

    rows = list(
        table.values()
    )

    rows.sort(
        key=lambda x: (
            x["points"],
            x["goals_for"]
            - x["goals_against"],
            x["goals_for"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        rows,
        start=1,
    ):

        row["rank"] = rank

        played = row["played"]

        row["goals_diff"] = (
            row["goals_for"]
            - row["goals_against"]
        )

        row["ppg"] = (
            round(
                row["points"]
                / played,
                3,
            )
            if played
            else 0
        )

    return rows


def _find_standing(
    standings,
    team_name,
):

    target = _normalize_team_name(
        team_name
    )

    for row in standings:

        if (
            _normalize_team_name(
                row["team"]
            )
            == target
        ):

            return row

    return None


# ================================================================
# FORM
# ================================================================

def _calculate_form(
    matches,
    team_name,
    before_date=None,
    limit=5,
):

    target = _normalize_team_name(
        team_name
    )

    matches = []

    for match in matches:

        date = match.get(
            "date"
        )

        if not date:
            continue

        if before_date and date >= before_date:
            continue

        score = _extract_score(
            match
        )

        if score is None:
            continue

        team1 = match.get(
            "team1",
            ""
        )

        team2 = match.get(
            "team2",
            ""
        )

        n1 = _normalize_team_name(
            team1
        )

        n2 = _normalize_team_name(
            team2
        )

        if (
            target != n1
            and
            target != n2
        ):
            continue

        matches.append(match)

    matches.sort(
        key=lambda x:
            x.get("date", ""),
        reverse=True,
    )

    matches = matches[:limit]

    results = []

    for match in matches:

        g1, g2 = _extract_score(
            match
        )

        team1 = match.get(
            "team1",
            ""
        )

        is_home = (
            _normalize_team_name(
                team1
            )
            == target
        )

        gf, ga = (
            (g1, g2)
            if is_home
            else
            (g2, g1)
        )

        if gf > ga:
            result = "W"
            points = 3

        elif gf == ga:
            result = "D"
            points = 1

        else:
            result = "L"
            points = 0

        results.append({

            "date":
                match.get(
                    "date"
                ),

            "opponent":
                (
                    match.get(
                        "team2"
                    )
                    if is_home
                    else
                    match.get(
                        "team1"
                    )
                ),

            "venue":
                "home"
                if is_home
                else
                "away",

            "result":
                result,

            "points":
                points,

            "goals_for":
                gf,

            "goals_against":
                ga,
        })

    form_string = "".join(
        x["result"]
        for x in reversed(results)
    )

    points = sum(
        x["points"]
        for x in results
    )

    return {

        "string":
            form_string,

        "last5":
            results,

        "points":
            points,

        "ppg":
            round(
                points / len(results),
                3,
            )
            if results
            else 0,
    }


# ================================================================
# H2H FROM OPENFOOTBALL
# ================================================================

def _calculate_h2h(
    matches,
    home_team,
    away_team,
    before_date=None,
    limit=10,
):

    home_norm = _normalize_team_name(
        home_team
    )

    away_norm = _normalize_team_name(
        away_team
    )

    h2h = []

    for match in matches:

        date = match.get(
            "date"
        )

        if not date:
            continue

        if before_date and date >= before_date:
            continue

        score = _extract_score(
            match
        )

        if score is None:
            continue

        t1 = _normalize_team_name(
            match.get(
                "team1",
                ""
            )
        )

        t2 = _normalize_team_name(
            match.get(
                "team2",
                ""
            )
        )

        if not (
            (
                t1 == home_norm
                and
                t2 == away_norm
            )
            or
            (
                t1 == away_norm
                and
                t2 == home_norm
            )
        ):

            continue

        h2h.append(match)

    h2h.sort(
        key=lambda x:
            x.get("date", ""),
        reverse=True,
    )

    h2h = h2h[:limit]

    home_wins = 0
    draws = 0
    away_wins = 0

    home_goals = 0
    away_goals = 0

    history = []

    for match in h2h:

        g1, g2 = _extract_score(
            match
        )

        t1 = _normalize_team_name(
            match.get(
                "team1",
                ""
            )
        )

        if t1 == home_norm:

            hg = g1
            ag = g2

        else:

            hg = g2
            ag = g1

        home_goals += hg
        away_goals += ag

        if hg > ag:
            home_wins += 1
            result = "HOME_WIN"

        elif hg == ag:
            draws += 1
            result = "DRAW"

        else:
            away_wins += 1
            result = "AWAY_WIN"

        history.append({

            "date":
                match.get("date"),

            "home_team":
                match.get("team1"),

            "away_team":
                match.get("team2"),

            "score":
                f"{g1}-{g2}",

            "home_perspective_score":
                f"{hg}-{ag}",

            "result":
                result,
        })

    total = len(h2h)

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
            round(
                home_wins / total,
                4,
            )
            if total
            else None,

        "draw_pct":
            round(
                draws / total,
                4,
            )
            if total
            else None,

        "away_win_pct":
            round(
                away_wins / total,
                4,
            )
            if total
            else None,

        "history":
            history,
    }


# ================================================================
# CURRENT CONTEXT
# ================================================================

def _build_current_context(
    matches,
    home_team,
    away_team,
    match_date,
):

    standings = _calculate_standings(
        matches,
        before_date=match_date,
    )

    home_standing = _find_standing(
        standings,
        home_team,
    )

    away_standing = _find_standing(
        standings,
        away_team,
    )

    home_form = _calculate_form(
        matches,
        home_team,
        before_date=match_date,
        limit=5,
    )

    away_form = _calculate_form(
        matches,
        away_team,
        before_date=match_date,
        limit=5,
    )

    h2h = _calculate_h2h(
        matches,
        home_team,
        away_team,
        before_date=match_date,
        limit=10,
    )

    return {

        "standings": {

            "home":
                home_standing,

            "away":
                away_standing,
        },

        "form": {

            "home":
                home_form,

            "away":
                away_form,
        },

        "h2h":
            h2h,

        "source":
            "OpenFootball",

        "odds":
            None,
    }


# ================================================================
# CONFIDENCE
# ================================================================

def _clamp(
    value,
    minimum=0,
    maximum=100,
):

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def _confidence_label(
    score,
):

    if score >= 78:
        return "TRÈS FORT"

    if score >= 68:
        return "FORT"

    if score >= 58:
        return "MOYEN"

    return "FAIBLE"


def _calculate_confidence(
    prediction,
    current_context,
):

    goals = prediction.get(
        "goals",
        {}
    )

    p_home = float(
        goals.get(
            "home_win",
            0,
        )
    )

    p_draw = float(
        goals.get(
            "draw",
            0,
        )
    )

    p_away = float(
        goals.get(
            "away_win",
            0,
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

    ordered = sorted(
        probs.values(),
        reverse=True,
    )

    top_probability = ordered[0]
    second_probability = ordered[1]

    margin = (
        top_probability
        - second_probability
    )

    score = (
        45
        + max(
            0,
            top_probability * 100 - 33,
        ) * 0.55
        + max(
            0,
            margin * 100,
        ) * 0.35
    )

    evidence = []

    standings = current_context.get(
        "standings",
        {}
    )

    home_standing = standings.get(
        "home"
    )

    away_standing = standings.get(
        "away"
    )

    if (
        home_standing
        and
        away_standing
    ):

        hp = home_standing.get(
            "points",
            0,
        )

        ap = away_standing.get(
            "points",
            0,
        )

        if predicted_side == "home" and hp > ap:

            score += 5

            evidence.append(
                "Classement favorable au domicile"
            )

        elif predicted_side == "away" and ap > hp:

            score += 5

            evidence.append(
                "Classement favorable à l'extérieur"
            )

    forms = current_context.get(
        "form",
        {}
    )

    hf = forms.get(
        "home",
        {}
    ).get(
        "ppg",
        0,
    )

    af = forms.get(
        "away",
        {}
    ).get(
        "ppg",
        0,
    )

    if predicted_side == "home" and hf > af:

        score += 6

        evidence.append(
            "Forme récente favorable"
        )

    elif predicted_side == "away" and af > hf:

        score += 6

        evidence.append(
            "Forme récente favorable"
        )

    h2h = current_context.get(
        "h2h",
        {}
    )

    if h2h.get("matches", 0) >= 3:

        if (
            predicted_side == "home"
            and
            (h2h.get(
                "home_win_pct"
            ) or 0) >= 0.50
        ):

            score += 4

            evidence.append(
                "Historique H2H favorable"
            )

        elif (
            predicted_side == "away"
            and
            (h2h.get(
                "away_win_pct"
            ) or 0) >= 0.50
        ):

            score += 4

            evidence.append(
                "Historique H2H favorable"
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
                top_probability * 100,
                2,
            ),

        "margin":
            round(
                margin * 100,
                2,
            ),

        "evidence":
            evidence[:5],
    }


# ================================================================
# BEST PICK
# ================================================================

def _build_best_pick(
    prediction,
):

    goals = prediction.get(
        "goals",
        {}
    )

    if not goals:
        return None

    candidates = [

        {
            "market": "1",
            "probability":
                float(
                    goals.get(
                        "home_win",
                        0,
                    )
                ),
            "label":
                "Victoire domicile",
        },

        {
            "market": "X",
            "probability":
                float(
                    goals.get(
                        "draw",
                        0,
                    )
                ),
            "label":
                "Match nul",
        },

        {
            "market": "2",
            "probability":
                float(
                    goals.get(
                        "away_win",
                        0,
                    )
                ),
            "label":
                "Victoire extérieur",
        },
    ]

    ou = goals.get(
        "over_under",
        {}
    )

    for line in [
        "1.5",
        "2.5",
        "3.5",
    ]:

        if line not in ou:
            continue

        candidates.append({

            "market":
                f"Over {line}",

            "probability":
                float(
                    ou[line].get(
                        "over",
                        0,
                    )
                ),

            "label":
                f"Plus de {line} buts",
        })

        candidates.append({

            "market":
                f"Under {line}",

            "probability":
                float(
                    ou[line].get(
                        "under",
                        0,
                    )
                ),

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
                best["probability"] * 100,
                2,
            ),
    }


# ================================================================
# JSON CLEANER
# ================================================================

def _clean_json(obj):

    if isinstance(obj, dict):

        return {
            str(k):
                _clean_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):

        return [
            _clean_json(v)
            for v in obj
        ]

    if isinstance(obj, tuple):

        return [
            _clean_json(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):

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


# ================================================================
# SINGLE MATCH
# ================================================================

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
                "GAGNE TEMPS predictor "
                "is not loaded."
            ),
        )

    try:

        match_date = pd.Timestamp(
            req.match_date
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid match_date.",
        )

    home_team = req.home_team.strip()
    away_team = req.away_team.strip()
    league = req.league.strip().upper()
    season = req.season.strip()

    team_index = _build_team_index()

    mapped_home = _match_dataset_team(
        home_team,
        team_index,
    )

    mapped_away = _match_dataset_team(
        away_team,
        team_index,
    )

    if not mapped_home:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown home team: {home_team}"
            ),
        )

    if not mapped_away:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown away team: {away_team}"
            ),
        )

    if league not in SUPPORTED_LEAGUES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported league: {league}"
            ),
        )

    t0 = time.time()

    result = club_predictor.predict(
        home_team=mapped_home,
        away_team=mapped_away,
        league=league,
        season=season,
        match_date=match_date,
        week=req.week,
    )

    elapsed = round(
        (time.time() - t0) * 1000,
        1,
    )

    return _clean_json({

        "status":
            "success",

        "match": {

            "home_team":
                home_team,

            "away_team":
                away_team,

            "tika_home_team":
                mapped_home,

            "tika_away_team":
                mapped_away,

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
                "TikaML historical features",

            "elapsed_ms":
                elapsed,
        },
    })


# ================================================================
# TODAY
# ================================================================

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

    fixture_date = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d"
    )

    team_index = _build_team_index()

    results = []
    skipped = []

    total_fixtures = 0

    # ------------------------------------------------------------
    # LOAD EACH SUPPORTED LEAGUE
    # ------------------------------------------------------------

    for tika_code, league_info in (
        SUPPORTED_LEAGUES.items()
    ):

        try:

            data = _get_openfootball_dataset(
                tika_code,
                str(
                    datetime.now(
                        timezone.utc
                    ).year
                ),
            )

        except Exception as e:

            log.exception(
                f"OpenFootball failed "
                f"for {tika_code}"
            )

            skipped.append({

                "league":
                    league_info["name"],

                "reason":
                    str(e),
            })

            continue

        matches = data.get(
            "matches",
            []
        )

        total_fixtures += len(matches)

        # --------------------------------------------------------
        # TODAY'S MATCHES
        # --------------------------------------------------------

        today_matches = [

            match

            for match in matches

            if match.get(
                "date"
            ) == fixture_date
        ]

        if not today_matches:

            continue

        for match in today_matches:

            try:

                api_home = match.get(
                    "team1"
                )

                api_away = match.get(
                    "team2"
                )

                home_team = _match_dataset_team(
                    api_home,
                    team_index,
                )

                away_team = _match_dataset_team(
                    api_away,
                    team_index,
                )

                if (
                    not home_team
                    or
                    not away_team
                ):

                    skipped.append({

                        "league":
                            league_info["name"],

                        "home_team":
                            api_home,

                        "away_team":
                            api_away,

                        "reason":
                            "Team name not found "
                            "in TikaML dataset.",
                    })

                    continue

                # ------------------------------------------------
                # CURRENT CONTEXT
                # ------------------------------------------------

                context = _build_current_context(
                    matches,
                    api_home,
                    api_away,
                    fixture_date,
                )

                # ------------------------------------------------
                # ROUND
                # ------------------------------------------------

                round_name = match.get(
                    "round"
                )

                week_number = None

                if round_name:

                    found = re.search(
                        r"(\d+)",
                        str(round_name),
                    )

                    if found:

                        week_number = int(
                            found.group(1)
                        )

                # ------------------------------------------------
                # TIKAML
                # ------------------------------------------------

                t0 = time.time()

                prediction = club_predictor.predict(
                    home_team=home_team,
                    away_team=away_team,
                    league=tika_code,
                    season=str(
                        datetime.now(
                            timezone.utc
                        ).year
                    ),
                    match_date=pd.Timestamp(
                        fixture_date
                    ),
                    week=week_number,
                    odds=None,
                )

                elapsed = round(
                    (time.time() - t0) * 1000,
                    1,
                )

                prediction = _clean_json(
                    prediction
                )

                # ------------------------------------------------
                # BEST PICK
                # ------------------------------------------------

                best_pick = _build_best_pick(
                    prediction
                )

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = _calculate_confidence(
                    prediction,
                    context,
                )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                result = {

                    "fixture_id":
                        f"{tika_code}-"
                        f"{fixture_date}-"
                        f"{_normalize_team_name(api_home)}-"
                        f"{_normalize_team_name(api_away)}",

                    "status":
                        (
                            "FINISHED"
                            if _match_is_finished(match)
                            else
                            "SCHEDULED"
                        ),

                    "kickoff":
                        match.get(
                            "time"
                        ),

                    "match": {

                        "home_team":
                            api_home,

                        "away_team":
                            api_away,

                        "tika_home_team":
                            home_team,

                        "tika_away_team":
                            away_team,
                    },

                    "league": {

                        "name":
                            league_info["name"],

                        "tika_code":
                            tika_code,

                        "openfootball":
                            league_info[
                                "openfootball_code"
                            ],
                    },

                    "season":
                        str(
                            datetime.now(
                                timezone.utc
                            ).year
                        ),

                    "week":
                        week_number,

                    "prediction":
                        prediction,

                    "best_pick":
                        best_pick,

                    "confidence":
                        confidence,

                    "current_context":
                        context,

                    "model_metadata": {

                        "engine":
                            "TikaML MatchPredictor",

                        "version":
                            models.version,

                        "source":
                            "OpenFootball + TikaML",

                        "odds":
                            False,

                        "standings":
                            True,

                        "form":
                            True,

                        "h2h":
                            True,

                        "elapsed_ms":
                            elapsed,
                    },
                }

                results.append(
                    result
                )

            except Exception as e:

                log.exception(
                    "GAGNE TEMPS prediction failed"
                )

                skipped.append({

                    "league":
                        league_info["name"],

                    "home_team":
                        match.get(
                            "team1"
                        ),

                    "away_team":
                        match.get(
                            "team2"
                        ),

                    "reason":
                        str(e),
                })

    # ------------------------------------------------------------
    # SORT BY CONFIDENCE
    # ------------------------------------------------------------

    results.sort(
        key=lambda item:
            item.get(
                "confidence",
                {}
            ).get(
                "score",
                0,
            ),
        reverse=True,
    )

    # ------------------------------------------------------------
    # TOP 5
    # ------------------------------------------------------------

    top5 = []

    for rank, item in enumerate(
        results[:5],
        start=1,
    ):

        goals = item.get(
            "prediction",
            {}
        ).get(
            "goals",
            {}
        )

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
                goals.get(
                    "recommended_score"
                ),
        })

    return _clean_json({

        "status":
            "success",

        "date":
            fixture_date,

        "source":
            "OpenFootball + TikaML",

        "model":
            "TikaML MatchPredictor",

        "version":
            models.version,

        "supported_leagues":
            [
                x["name"]
                for x in SUPPORTED_LEAGUES.values()
            ],

        "total_openfootball_matches":
            total_fixtures,

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

        "odds":
            {
                "available":
                    False,

                "reason":
                    "OpenFootball does not provide bookmaker odds.",
            },

        "disclaimer":
            (
                "Predictions are statistical estimates "
                "and are not guarantees."
            ),
    })


# ================================================================
# TOP 5
# ================================================================

@app.get(
    "/gagne-temps/top",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_top():

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


# ================================================================
# STANDARD /predict
# ================================================================

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
        req.prediction_type == "live"
        and req.match_context
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
        (time.time() - t0) * 1000,
        1,
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


# ================================================================
# BACKFILL
# ================================================================

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
        (time.time() - t0) * 1000,
        1,
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


# ================================================================
# MODEL STATUS
# ================================================================

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

        "football_data": {

            "provider":
                "OpenFootball",

            "api_key_required":
                False,

            "source":
                "GitHub raw JSON",

            "repository":
                "openfootball/football.json",

            "current_season":
                "2026-27",

            "leagues":
                [
                    {
                        "code":
                            code,

                        "name":
                            league["name"],

                        "file":
                            league["file"],
                    }

                    for code, league
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
