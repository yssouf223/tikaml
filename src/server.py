"""
GAGNE TEMPS / TikaML Prediction API
-----------------------------------

Serveur FastAPI pour :
- prédictions TikaML classiques
- prédictions live
- backfill
- équipes nationales
- GAGNE TEMPS
- matchs du jour via OpenFootball
- Top 5 des meilleurs pronostics

Football data provider:
    OpenFootball / football.json

Modèle:
    TikaML LightGBM + Poisson

IMPORTANT:
    Les clés secrètes doivent rester dans les variables
    d'environnement Render.
"""

import json
import logging
import math
import os
import re
import secrets
import time
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from pydantic import BaseModel, Field

from scipy.stats import poisson

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.lgbm_poisson import (
    LGBMPoissonModel,
)

from src.live_predictor import LivePredictor

from src import national_api

from src.inference import MatchPredictor


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("gagne-temps")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = Path("models/corners")
YELLOW_MODEL_DIR = Path("models/yellows")
BACKFILL_DIR = Path("models/backfill_20260131")

MAX_GOALS = 7

# OpenFootball
OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/"
    "football.json/master"
)

OPENFOOTBALL_SEASON = os.environ.get(
    "OPENFOOTBALL_SEASON",
    "2026-27",
)

OPENFOOTBALL_CACHE_SECONDS = int(
    os.environ.get(
        "OPENFOOTBALL_CACHE_SECONDS",
        "1800",
    )
)

SUPPORTED_LEAGUES = {
    "EPL": {
        "name": "Premier League",
        "file": "en.1.json",
        "openfootball": "en.1",
        "tika_code": "EPL",
        "country": "England",
    },
    "LL": {
        "name": "La Liga",
        "file": "es.1.json",
        "openfootball": "es.1",
        "tika_code": "LL",
        "country": "Spain",
    },
    "SEA": {
        "name": "Serie A",
        "file": "it.1.json",
        "openfootball": "it.1",
        "tika_code": "SEA",
        "country": "Italy",
    },
    "BUN": {
        "name": "Bundesliga",
        "file": "de.1.json",
        "openfootball": "de.1",
        "tika_code": "BUN",
        "country": "Germany",
    },
    "LI1": {
        "name": "Ligue 1",
        "file": "fr.1.json",
        "openfootball": "fr.1",
        "tika_code": "LI1",
        "country": "France",
    },
}


# ============================================================
# API KEY
# ============================================================

API_KEY = os.environ.get("TIKA_API_KEY", "").strip()

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(
        "TIKA_API_KEY non défini. Une clé temporaire a été générée."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str = Security(api_key_header),
):
    """
    Vérifie X-API-Key.
    """

    if not key:
        raise HTTPException(
            status_code=403,
            detail="API key required",
        )

    if not secrets.compare_digest(
        key,
        API_KEY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ============================================================
# MODELS
# ============================================================

class Models:
    goals: LGBMPoissonModel | None = None
    corners: LGBMPoissonModel | None = None
    yellows: LGBMPoissonModel | None = None
    version: str = "unknown"


models = Models()
backfill_models = Models()


# ============================================================
# GAGNE TEMPS MODEL
# ============================================================

club_predictor: MatchPredictor | None = None


# ============================================================
# OPENFOOTBALL CACHE
# ============================================================

OPENFOOTBALL_CACHE: dict[str, dict[str, Any]] = {}


# ============================================================
# TEAM ALIASES
# ============================================================

TEAM_ALIASES = {
    # Bundesliga
    "1. FC Union Berlin": [
        "Union Berlin",
        "1 FC Union Berlin",
        "Union",
    ],
    "FC Schalke 04": [
        "Schalke 04",
        "Schalke",
        "FC Schalke",
    ],
    "Bayern München": [
        "Bayern Munich",
        "Bayern",
        "FC Bayern München",
        "FC Bayern Munich",
    ],
    "Borussia Dortmund": [
        "Dortmund",
        "BVB",
    ],
    "Bayer 04 Leverkusen": [
        "Bayer Leverkusen",
        "Leverkusen",
    ],
    "RB Leipzig": [
        "Leipzig",
    ],

    # France
    "Stade Rennais FC 1901": [
        "Rennes",
        "Stade Rennais",
        "Stade Rennais FC",
    ],
    "Olympique de Marseille": [
        "Marseille",
        "Olympique Marseille",
        "Olympique de Marseille",
    ],
    "Paris Saint-Germain FC": [
        "Paris Saint-Germain",
        "PSG",
    ],
    "AS Monaco FC": [
        "Monaco",
        "AS Monaco",
    ],
    "Olympique Lyonnais": [
        "Lyon",
        "Olympique Lyon",
    ],
    "LOSC Lille Métropole": [
        "Lille",
        "LOSC Lille",
    ],

    # England
    "Manchester United FC": [
        "Manchester United",
        "Man United",
    ],
    "Manchester City FC": [
        "Manchester City",
        "Man City",
    ],
    "Liverpool FC": [
        "Liverpool",
    ],
    "Chelsea FC": [
        "Chelsea",
    ],
    "Arsenal FC": [
        "Arsenal",
    ],
    "Tottenham Hotspur FC": [
        "Tottenham",
        "Spurs",
    ],

    # Spain
    "Real Madrid CF": [
        "Real Madrid",
    ],
    "FC Barcelona": [
        "Barcelona",
        "Barca",
    ],
    "Club Atlético de Madrid": [
        "Atletico Madrid",
        "Atlético Madrid",
        "Atletico",
    ],
    "Sevilla FC": [
        "Sevilla",
    ],
    "Valencia CF": [
        "Valencia",
    ],

    # Italy
    "FC Internazionale Milano": [
        "Inter",
        "Internazionale",
        "Inter Milan",
    ],
    "AC Milan": [
        "Milan",
    ],
    "Juventus FC": [
        "Juventus",
    ],
    "SSC Napoli": [
        "Napoli",
    ],
    "AS Roma": [
        "Roma",
    ],
}


# ============================================================
# APP LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global club_predictor

    log.info("==========================================")
    log.info("Starting GAGNE TEMPS / TikaML")
    log.info("==========================================")

    start = time.time()

    # --------------------------------------------------------
    # MAIN GOALS MODEL
    # --------------------------------------------------------

    try:
        models.goals = LGBMPoissonModel.load(
            str(MODEL_DIR)
        )

        log.info(
            "Goals model loaded: %s features",
            len(models.goals.feature_cols),
        )

    except Exception as exc:
        log.exception(
            "Impossible de charger le modèle goals: %s",
            exc,
        )

    # --------------------------------------------------------
    # CORNERS
    # --------------------------------------------------------

    if CORNER_MODEL_DIR.exists():

        try:
            models.corners = LGBMPoissonModel.load(
                str(CORNER_MODEL_DIR)
            )

            log.info(
                "Corners model loaded: %s features",
                len(models.corners.feature_cols),
            )

        except Exception as exc:
            log.warning(
                "Corners model non chargé: %s",
                exc,
            )

    # --------------------------------------------------------
    # YELLOWS
    # --------------------------------------------------------

    if YELLOW_MODEL_DIR.exists():

        try:
            models.yellows = LGBMPoissonModel.load(
                str(YELLOW_MODEL_DIR)
            )

            log.info(
                "Yellows model loaded: %s features",
                len(models.yellows.feature_cols),
            )

        except Exception as exc:
            log.warning(
                "Yellows model non chargé: %s",
                exc,
            )

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    meta_path = MODEL_DIR / "meta.json"

    if meta_path.exists():

        try:

            with open(
                meta_path,
                "r",
                encoding="utf-8",
            ) as f:
                meta = json.load(f)

            models.version = (
                f"lgbm-poisson-"
                f"{len(meta.get('feature_cols', []))}f"
            )

        except Exception:
            models.version = "lgbm-poisson"

    # --------------------------------------------------------
    # BACKFILL
    # --------------------------------------------------------

    if BACKFILL_DIR.exists():

        try:

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

            log.info("Backfill models loaded")

        except Exception as exc:

            log.warning(
                "Backfill models non chargés: %s",
                exc,
            )

    # --------------------------------------------------------
    # GAGNE TEMPS MATCH PREDICTOR
    # --------------------------------------------------------

    try:

        club_predictor = MatchPredictor()

        club_predictor.load_model()

        log.info(
            "GAGNE TEMPS predictor loaded "
            "(%s historical matches)",
            len(club_predictor.df),
        )

    except Exception as exc:

        club_predictor = None

        log.exception(
            "GAGNE TEMPS predictor non chargé: %s",
            exc,
        )

    # --------------------------------------------------------
    # NATIONAL MODEL
    # --------------------------------------------------------

    try:

        nm = national_api.load_national()

        log.info(
            "National model loaded (%s teams)",
            len(nm.attack),
        )

    except Exception as exc:

        log.warning(
            "National model non chargé: %s",
            exc,
        )

    elapsed = time.time() - start

    log.info(
        "Tous les modèles chargés en %.1fs",
        elapsed,
    )

    yield

    log.info("Shutting down GAGNE TEMPS")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="GAGNE TEMPS — TikaML Football Prediction API",
    version="2.0.0",
    description=(
        "Football prediction API powered by TikaML "
        "and OpenFootball."
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# NATIONAL ROUTES
# ============================================================

app.include_router(
    national_api.router,
    dependencies=[
        Depends(verify_api_key)
    ],
)


# ============================================================
# REQUEST MODELS
# ============================================================

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

    feature_vector: dict[
        str,
        float | int | None
    ]

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


class GagneTempsRequest(BaseModel):

    home_team: str
    away_team: str

    league: str
    season: str

    match_date: str

    week: int | None = None


# ============================================================
# GENERIC HELPERS
# ============================================================

def _json_clean(value):

    """
    Transforme numpy/pandas/nan en JSON propre.
    """

    if isinstance(
        value,
        dict,
    ):

        return {
            str(k): _json_clean(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            _json_clean(v)
            for v in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return [
            _json_clean(v)
            for v in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):

        return [
            _json_clean(v)
            for v in value.tolist()
        ]

    if isinstance(
        value,
        (np.integer,),
    ):

        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):

        value = float(value)

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if isinstance(
        value,
        float,
    ):

        if math.isnan(value):
            return None

        if math.isinf(value):
            return None

        return value

    if pd.isna(value):
        return None

    return value


def _build_feature_df(
    feature_vector: dict,
    feature_list: list[str],
) -> pd.DataFrame:

    row = {}

    for col in feature_list:

        value = feature_vector.get(col)

        if value is None:
            row[col] = np.nan

        else:

            try:
                row[col] = float(value)

            except Exception:
                row[col] = np.nan

    return pd.DataFrame([row])


# ============================================================
# POISSON HELPERS
# ============================================================

def _goals_over_under(
    matrix: np.ndarray,
) -> dict:

    output = {}

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

        output[str(line)] = {
            "over": round(
                float(p_over),
                4,
            ),
            "under": round(
                float(1 - p_over),
                4,
            ),
        }

    return output


def _poisson_over_under(
    lambda_total: float,
    lines: list[float],
) -> dict:

    output = {}

    for line in lines:

        p_over = float(
            1
            - poisson.cdf(
                int(line),
                lambda_total,
            )
        )

        output[str(line)] = {
            "over": round(
                p_over,
                4,
            ),
            "under": round(
                1 - p_over,
                4,
            ),
        }

    return output


def _live_over_under(
    lambda_remaining: float,
    current_total: int,
    lines: list[float],
) -> dict:

    output = {}

    for line in lines:

        needed = line - current_total

        if needed <= 0:

            output[str(line)] = {
                "over": 1.0,
                "under": 0.0,
            }

            continue

        p_over = float(
            1
            - poisson.cdf(
                int(needed),
                lambda_remaining,
            )
        )

        output[str(line)] = {
            "over": round(
                p_over,
                4,
            ),
            "under": round(
                1 - p_over,
                4,
            ),
        }

    return output


# ============================================================
# PREMIATCH
# ============================================================

def predict_prematch(
    feature_vector: dict,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    # --------------------------------------------------------
    # GOALS
    # --------------------------------------------------------

    if (
        "goals" in requested_models
        and m.goals
    ):

        feat_df = _build_feature_df(
            feature_vector,
            m.goals.feature_cols,
        )

        lh, la = m.goals.predict_lambdas(
            feat_df
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
                    float(matrix[i, j]),
                    4,
                )
                for j in range(MAX_GOALS)
            ]
            for i in range(MAX_GOALS)
        ]

        best_i, best_j = divmod(
            int(
                np.argmax(matrix)
            ),
            MAX_GOALS,
        )

        recommended_score = {
            "home_goals": best_i,
            "away_goals": best_j,
            "prob": round(
                float(
                    matrix[
                        best_i,
                        best_j
                    ]
                ),
                4,
            ),
            "label": (
                f"{best_i}-{best_j}"
            ),
        }

        predictions["goals"] = {

            "home_win": round(
                p_home,
                4,
            ),

            "draw": round(
                p_draw,
                4,
            ),

            "away_win": round(
                p_away,
                4,
            ),

            "expected_home": round(
                lh,
                4,
            ),

            "expected_away": round(
                la,
                4,
            ),

            "predicted_total": round(
                lh + la,
                4,
            ),

            "over_under": (
                _goals_over_under(
                    matrix
                )
            ),

            "score_matrix": score_matrix,

            "recommended_score": (
                recommended_score
            ),
        }

    # --------------------------------------------------------
    # CORNERS
    # --------------------------------------------------------

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

            "expected_home": round(
                clh,
                4,
            ),

            "expected_away": round(
                cla,
                4,
            ),

            "predicted_total": round(
                clh + cla,
                4,
            ),

            "over_under": (
                _poisson_over_under(
                    clh + cla,
                    [
                        8.5,
                        9.5,
                        10.5,
                        11.5,
                    ],
                )
            ),
        }

    # --------------------------------------------------------
    # YELLOWS
    # --------------------------------------------------------

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

            "expected_home": round(
                ylh,
                4,
            ),

            "expected_away": round(
                yla,
                4,
            ),

            "predicted_total": round(
                ylh + yla,
                4,
            ),

            "over_under": (
                _poisson_over_under(
                    ylh + yla,
                    [
                        2.5,
                        3.5,
                        4.5,
                        5.5,
                    ],
                )
            ),
        }

    return predictions


# ============================================================
# LIVE
# ============================================================

def predict_live(
    feature_vector: dict,
    ctx: MatchContext,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    minute = ctx.minute or 0

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

        predictor = LivePredictor(
            lh,
            la,
            rho=m.goals.rho,
            max_goals=MAX_GOALS,
        )

        predictor.update(
            minute=minute,
            home_goals=ctx.home_score,
            away_goals=ctx.away_score,
            home_red_cards=ctx.home_red_cards,
            away_red_cards=ctx.away_red_cards,
        )

        live = predictor.get_probabilities()

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
                for j in range(MAX_GOALS)
            ]
            for i in range(MAX_GOALS)
        ]

        best_i, best_j = divmod(
            int(
                np.argmax(matrix)
            ),
            MAX_GOALS,
        )

        predictions["goals"] = {

            "home_win": round(
                float(
                    live[
                        "probs_1x2"
                    ][0]
                ),
                4,
            ),

            "draw": round(
                float(
                    live[
                        "probs_1x2"
                    ][1]
                ),
                4,
            ),

            "away_win": round(
                float(
                    live[
                        "probs_1x2"
                    ][2]
                ),
                4,
            ),

            "expected_home": round(
                lh,
                4,
            ),

            "expected_away": round(
                la,
                4,
            ),

            "predicted_total": round(
                lh + la,
                4,
            ),

            "over_under": {
                str(k): {
                    "over": round(
                        float(v["over"]),
                        4,
                    ),
                    "under": round(
                        float(v["under"]),
                        4,
                    ),
                }
                for k, v in live[
                    "over_under"
                ].items()
            },

            "lambda_remaining_home": round(
                float(
                    live[
                        "lambda_remaining"
                    ][0]
                ),
                4,
            ),

            "lambda_remaining_away": round(
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

            "score_matrix": score_matrix,

            "recommended_score": {
                "home_goals": best_i,
                "away_goals": best_j,
                "prob": round(
                    float(
                        matrix[
                            best_i,
                            best_j
                        ]
                    ),
                    4,
                ),
                "label": (
                    f"{best_i}-{best_j}"
                ),
            },
        }

    remaining_ratio = max(
        0,
        (90 - minute) / 90,
    )

    # Corners
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

            "expected_home": round(
                clh,
                4,
            ),

            "expected_away": round(
                cla,
                4,
            ),

            "predicted_total": round(
                clh + cla,
                4,
            ),

            "lambda_remaining_home": round(
                clh * remaining_ratio,
                4,
            ),

            "lambda_remaining_away": round(
                cla * remaining_ratio,
                4,
            ),

            "over_under": (
                _live_over_under(
                    (
                        clh
                        + cla
                    )
                    * remaining_ratio,
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
                )
            ),
        }

    # Yellows
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

            "expected_home": round(
                ylh,
                4,
            ),

            "expected_away": round(
                yla,
                4,
            ),

            "predicted_total": round(
                ylh + yla,
                4,
            ),

            "lambda_remaining_home": round(
                ylh * remaining_ratio,
                4,
            ),

            "lambda_remaining_away": round(
                yla * remaining_ratio,
                4,
            ),

            "over_under": (
                _live_over_under(
                    (
                        ylh
                        + yla
                    )
                    * remaining_ratio,
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
                )
            ),
        }

    return predictions


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_team_name(
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

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _get_tika_team_names() -> list[str]:

    if club_predictor is None:
        return []

    if club_predictor.df is None:
        return []

    if "home_team" not in club_predictor.df.columns:
        return []

    teams = set(
        club_predictor.df[
            "home_team"
        ].dropna().astype(str)
    )

    if "away_team" in club_predictor.df.columns:

        teams.update(
            club_predictor.df[
                "away_team"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

    return sorted(teams)


def resolve_tika_team(
    openfootball_name: str,
) -> str | None:

    """
    Convertit un nom OpenFootball vers le nom
    réellement présent dans features.csv.

    On ne fait jamais une correspondance arbitraire :
    le résultat doit exister dans le dataset TikaML.
    """

    tika_teams = _get_tika_team_names()

    if not tika_teams:
        return None

    normalized_index = {
        normalize_team_name(team): team
        for team in tika_teams
    }

    # 1. Match exact normalisé
    normalized = normalize_team_name(
        openfootball_name
    )

    if normalized in normalized_index:
        return normalized_index[
            normalized
        ]

    # 2. Alias explicites
    candidates = TEAM_ALIASES.get(
        openfootball_name,
        [],
    )

    candidates = (
        candidates
        + [openfootball_name]
    )

    for candidate in candidates:

        key = normalize_team_name(
            candidate
        )

        if key in normalized_index:
            return normalized_index[key]

    # 3. Heuristique prudente :
    # recherche d'une correspondance par inclusion
    # uniquement si elle est unique.
    matches = []

    for team in tika_teams:

        team_key = normalize_team_name(
            team
        )

        if (
            normalized in team_key
            or team_key in normalized
        ):

            matches.append(team)

    if len(matches) == 1:
        return matches[0]

    return None


# ============================================================
# OPENFOOTBALL HTTP
# ============================================================

def _http_get_json(
    url: str,
) -> dict:

    request = Request(
        url,
        headers={
            "User-Agent": (
                "GAGNE-TEMPS/2.0 "
                "(football prediction service)"
            ),
            "Accept": "application/json",
        },
    )

    try:

        with urlopen(
            request,
            timeout=20,
        ) as response:

            raw = response.read()

        return json.loads(
            raw.decode(
                "utf-8"
            )
        )

    except HTTPError as exc:

        raise RuntimeError(
            f"OpenFootball HTTP {exc.code}"
        ) from exc

    except URLError as exc:

        raise RuntimeError(
            f"OpenFootball network error: {exc}"
        ) from exc

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "OpenFootball JSON invalide"
        ) from exc


def load_openfootball_league(
    league_code: str,
) -> dict:

    if league_code not in SUPPORTED_LEAGUES:

        raise ValueError(
            f"Ligue inconnue: {league_code}"
        )

    cached = OPENFOOTBALL_CACHE.get(
        league_code
    )

    now = time.time()

    if cached:

        age = now - cached["timestamp"]

        if age < OPENFOOTBALL_CACHE_SECONDS:

            return cached["data"]

    info = SUPPORTED_LEAGUES[
        league_code
    ]

    url = (
        f"{OPENFOOTBALL_BASE}/"
        f"{OPENFOOTBALL_SEASON}/"
        f"{info['file']}"
    )

    data = _http_get_json(url)

    OPENFOOTBALL_CACHE[
        league_code
    ] = {
        "timestamp": now,
        "data": data,
        "url": url,
    }

    return data


def load_all_openfootball():

    result = {}

    errors = {}

    for league_code in SUPPORTED_LEAGUES:

        try:

            result[
                league_code
            ] = load_openfootball_league(
                league_code
            )

        except Exception as exc:

            errors[
                league_code
            ] = str(exc)

            log.warning(
                "OpenFootball %s failed: %s",
                league_code,
                exc,
            )

    return result, errors


# ============================================================
# OPENFOOTBALL MATCH HELPERS
# ============================================================

def parse_match_date(
    value: Any,
) -> str | None:

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):

        return value.strftime(
            "%Y-%m-%d"
        )

    if isinstance(
        value,
        date,
    ):

        return value.strftime(
            "%Y-%m-%d"
        )

    text = str(value).strip()

    if not text:
        return None

    # ISO date
    match = re.match(
        r"^(\d{4}-\d{2}-\d{2})",
        text,
    )

    if match:
        return match.group(1)

    # DD.MM.YYYY
    match = re.match(
        r"^(\d{2})\.(\d{2})\.(\d{4})",
        text,
    )

    if match:

        return (
            f"{match.group(3)}-"
            f"{match.group(2)}-"
            f"{match.group(1)}"
        )

    return None


def get_match_score(
    match: dict,
):

    score = match.get(
        "score"
    )

    if not isinstance(
        score,
        dict,
    ):
        return None

    ft = score.get(
        "ft"
    )

    if ft is None:
        return None

    if isinstance(
        ft,
        dict,
    ):

        home = (
            ft.get("1")
            or ft.get("home")
        )

        away = (
            ft.get("2")
            or ft.get("away")
        )

        if (
            home is None
            or away is None
        ):
            return None

        try:

            return (
                int(home),
                int(away),
            )

        except Exception:
            return None

    if isinstance(
        ft,
        list,
    ) and len(ft) >= 2:

        try:

            return (
                int(ft[0]),
                int(ft[1]),
            )

        except Exception:
            return None

    if isinstance(
        ft,
        str,
    ):

        match_score = re.search(
            r"(\d+)\s*[-:]\s*(\d+)",
            ft,
        )

        if match_score:

            return (
                int(
                    match_score.group(1)
                ),
                int(
                    match_score.group(2)
                ),
            )

    return None


def is_completed_match(
    match: dict,
) -> bool:

    return (
        get_match_score(match)
        is not None
    )


def get_match_round(
    match: dict,
) -> int | None:

    value = match.get(
        "round"
    )

    if value is None:
        return None

    text = str(value)

    match_number = re.search(
        r"(\d+)",
        text,
    )

    if not match_number:
        return None

    try:
        return int(
            match_number.group(1)
        )

    except Exception:
        return None


# ============================================================
# CURRENT STANDINGS
# ============================================================

def build_standings(
    matches: list[dict],
) -> dict:

    table = {}

    for match in matches:

        score = get_match_score(
            match
        )

        if score is None:
            continue

        home = match.get(
            "team1"
        )

        away = match.get(
            "team2"
        )

        if not home or not away:
            continue

        hg, ag = score

        if home not in table:

            table[home] = {
                "team": home,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_diff": 0,
                "points": 0,
            }

        if away not in table:

            table[away] = {
                "team": away,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_diff": 0,
                "points": 0,
            }

        h = table[home]
        a = table[away]

        h["played"] += 1
        a["played"] += 1

        h["goals_for"] += hg
        h["goals_against"] += ag

        a["goals_for"] += ag
        a["goals_against"] += hg

        if hg > ag:

            h["wins"] += 1
            a["losses"] += 1

            h["points"] += 3

        elif hg < ag:

            a["wins"] += 1
            h["losses"] += 1

            a["points"] += 3

        else:

            h["draws"] += 1
            a["draws"] += 1

            h["points"] += 1
            a["points"] += 1

    for team in table.values():

        team["goal_diff"] = (
            team["goals_for"]
            - team["goals_against"]
        )

    ordered = sorted(
        table.values(),
        key=lambda x: (
            -x["points"],
            -x["goal_diff"],
            -x["goals_for"],
        ),
    )

    output = {}

    for index, row in enumerate(
        ordered,
        start=1,
    ):

        row["position"] = index

        output[
            row["team"]
        ] = row

    return output


# ============================================================
# CURRENT FORM
# ============================================================

def get_team_form(
    matches: list[dict],
    team: str,
    limit: int = 5,
) -> dict:

    team_matches = []

    for match in matches:

        score = get_match_score(
            match
        )

        if score is None:
            continue

        home = match.get(
            "team1"
        )

        away = match.get(
            "team2"
        )

        if team not in (
            home,
            away,
        ):
            continue

        hg, ag = score

        if home == team:

            gf = hg
            ga = ag

        else:

            gf = ag
            ga = hg

        if gf > ga:
            result = "W"
            points = 3

        elif gf < ga:
            result = "L"
            points = 0

        else:
            result = "D"
            points = 1

        team_matches.append(
            {
                "date": (
                    parse_match_date(
                        match.get("date")
                    )
                ),
                "opponent": (
                    away
                    if home == team
                    else home
                ),
                "result": result,
                "points": points,
                "goals_for": gf,
                "goals_against": ga,
            }
        )

    team_matches.sort(
        key=lambda x: (
            x["date"] or ""
        )
    )

    recent = team_matches[
        -limit:
    ]

    return {
        "last_5": "".join(
            x["result"]
            for x in recent
        ),
        "points_last_5": sum(
            x["points"]
            for x in recent
        ),
        "goals_for_last_5": sum(
            x["goals_for"]
            for x in recent
        ),
        "goals_against_last_5": sum(
            x["goals_against"]
            for x in recent
        ),
        "matches": recent,
    }


# ============================================================
# H2H OPENFOOTBALL
# ============================================================

def get_openfootball_h2h(
    matches: list[dict],
    home_team: str,
    away_team: str,
    limit: int = 5,
) -> dict:

    h2h = []

    for match in matches:

        score = get_match_score(
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

        if {
            team1,
            team2,
        } != {
            home_team,
            away_team,
        }:
            continue

        hg, ag = score

        if team1 == home_team:

            home_goals = hg
            away_goals = ag

        else:

            home_goals = ag
            away_goals = hg

        if home_goals > away_goals:
            result = "W"

        elif home_goals < away_goals:
            result = "L"

        else:
            result = "D"

        h2h.append(
            {
                "date": (
                    parse_match_date(
                        match.get("date")
                    )
                ),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result_home": result,
            }
        )

    h2h.sort(
        key=lambda x: (
            x["date"] or ""
        )
    )

    return {
        "matches": h2h[-limit:],
        "count": len(h2h),
    }


# ============================================================
# CURRENT CONTEXT
# ============================================================

def build_current_context(
    matches: list[dict],
    home_team: str,
    away_team: str,
) -> dict:

    standings = build_standings(
        matches
    )

    home_table = standings.get(
        home_team
    )

    away_table = standings.get(
        away_team
    )

    home_form = get_team_form(
        matches,
        home_team,
    )

    away_form = get_team_form(
        matches,
        away_team,
    )

    h2h = get_openfootball_h2h(
        matches,
        home_team,
        away_team,
    )

    points_diff = 0
    position_diff = 0

    if home_table and away_table:

        points_diff = (
            home_table["points"]
            - away_table["points"]
        )

        # TikaML :
        # position_diff = away_pos - home_pos
        position_diff = (
            away_table["position"]
            - home_table["position"]
        )

    return {
        "standings": {
            "home": home_table,
            "away": away_table,
            "points_diff": points_diff,
            "position_diff": position_diff,
        },

        "form": {
            "home": home_form,
            "away": away_form,
        },

        "h2h": h2h,
    }


# ============================================================
# SAFE FEATURE OVERRIDE
# ============================================================

def apply_current_context_to_features(
    feature_vector: dict,
    context: dict,
):

    """
    Injecte uniquement les features connues de TikaML.
    """

    standings = context.get(
        "standings",
        {}
    )

    home = standings.get(
        "home"
    )

    away = standings.get(
        "away"
    )

    if home and away:

        safe_values = {
            "points_diff": (
                home["points"]
                - away["points"]
            ),

            "position_diff": (
                away["position"]
                - home["position"]
            ),
        }

        # ----------------------------------------------------
        # Relegation / title context
        # ----------------------------------------------------

        all_positions = []

        if home:
            all_positions.append(home)

        if away:
            all_positions.append(away)

        for key, value in safe_values.items():

            if key in feature_vector:
                feature_vector[key] = value

    # --------------------------------------------------------
    # H2H
    # --------------------------------------------------------

    h2h = context.get(
        "h2h",
        {}
    )

    h2h_matches = h2h.get(
        "matches",
        []
    )

    if h2h_matches:

        wins = 0
        goal_diff = 0

        for item in h2h_matches:

            hg = item[
                "home_goals"
            ]

            ag = item[
                "away_goals"
            ]

            goal_diff += (
                hg - ag
            )

            if hg > ag:
                wins += 1

            elif hg == ag:
                wins += 0.5

        n = len(
            h2h_matches
        )

        h2h_values = {
            "h2h_win_pct_home": (
                wins / n
            ),

            "h2h_goal_diff_home": (
                goal_diff / n
            ),

            "h2h_matches": n,
        }

        for key, value in (
            h2h_values.items()
        ):

            if key in feature_vector:
                feature_vector[key] = value

    return feature_vector


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    goals_prediction: dict,
    context: dict,
) -> dict:

    if not goals_prediction:

        return {
            "score": 0,
            "level": "FAIBLE",
            "predicted_side": None,
            "model_probability": 0,
            "margin": 0,
            "evidence": [],
        }

    probabilities = {
        "home": float(
            goals_prediction.get(
                "home_win",
                0,
            )
        ),

        "draw": float(
            goals_prediction.get(
                "draw",
                0,
            )
        ),

        "away": float(
            goals_prediction.get(
                "away_win",
                0,
            )
        ),
    }

    ordered = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    predicted_side = ordered[0][0]

    top_probability = ordered[0][1]

    second_probability = ordered[1][1]

    margin = (
        top_probability
        - second_probability
    )

    # --------------------------------------------------------
    # Base confidence
    # --------------------------------------------------------

    score = (
        top_probability * 100
    )

    # Margin bonus
    score += (
        margin * 45
    )

    evidence = []

    # --------------------------------------------------------
    # Form agreement
    # --------------------------------------------------------

    form = context.get(
        "form",
        {}
    )

    home_form = form.get(
        "home",
        {}
    )

    away_form = form.get(
        "away",
        {}
    )

    home_points = home_form.get(
        "points_last_5",
        0,
    )

    away_points = away_form.get(
        "points_last_5",
        0,
    )

    if predicted_side == "home":

        if home_points > away_points:
            score += 8
            evidence.append(
                "forme récente favorable à domicile"
            )

    elif predicted_side == "away":

        if away_points > home_points:
            score += 8
            evidence.append(
                "forme récente favorable à l'extérieur"
            )

    else:

        if abs(
            home_points
            - away_points
        ) <= 2:

            score += 6

            evidence.append(
                "formes récentes équilibrées"
            )

    # --------------------------------------------------------
    # Standings agreement
    # --------------------------------------------------------

    standings = context.get(
        "standings",
        {}
    )

    home_table = standings.get(
        "home"
    )

    away_table = standings.get(
        "away"
    )

    if (
        home_table
        and away_table
    ):

        if (
            predicted_side == "home"
            and home_table["position"]
            < away_table["position"]
        ):

            score += 7

            evidence.append(
                "classement favorable à domicile"
            )

        elif (
            predicted_side == "away"
            and away_table["position"]
            < home_table["position"]
        ):

            score += 7

            evidence.append(
                "classement favorable à l'extérieur"
            )

        elif (
            predicted_side == "draw"
            and abs(
                home_table["position"]
                - away_table["position"]
            ) <= 2
        ):

            score += 5

            evidence.append(
                "classements très proches"
            )

    # --------------------------------------------------------
    # H2H
    # --------------------------------------------------------

    h2h = context.get(
        "h2h",
        {}
    )

    h2h_matches = h2h.get(
        "matches",
        []
    )

    if len(h2h_matches) >= 3:

        home_h2h_wins = sum(
            1
            for item in h2h_matches
            if item["result_home"]
            == "W"
        )

        if (
            predicted_side == "home"
            and home_h2h_wins
            >= len(h2h_matches) / 2
        ):

            score += 3

            evidence.append(
                "historique H2H favorable à domicile"
            )

        elif (
            predicted_side == "away"
            and home_h2h_wins
            <= len(h2h_matches) / 3
        ):

            score += 3

            evidence.append(
                "historique H2H favorable à l'extérieur"
            )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    if score >= 75:
        level = "FORTE"

    elif score >= 58:
        level = "MOYENNE"

    else:
        level = "FAIBLE"

    return {
        "score": round(
            score,
            1,
        ),

        "level": level,

        "predicted_side": (
            predicted_side
        ),

        "model_probability": round(
            top_probability,
            4,
        ),

        "margin": round(
            margin,
            4,
        ),

        "evidence": evidence[:5],
    }


# ============================================================
# GAGNE TEMPS PREDICTION
# ============================================================

def gagne_temps_predict(
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    match_date: str,
    week: int | None = None,
    openfootball_matches: list[dict] | None = None,
):

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "GAGNE TEMPS predictor "
                "not loaded"
            ),
        )

    if league not in SUPPORTED_LEAGUES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Ligue non supportée: {league}. "
                f"Valeurs: "
                f"{list(SUPPORTED_LEAGUES.keys())}"
            ),
        )

    tika_home = resolve_tika_team(
        home_team
    )

    tika_away = resolve_tika_team(
        away_team
    )

    if not tika_home:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Équipe domicile introuvable "
                f"dans TikaML: {home_team}"
            ),
        )

    if not tika_away:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Équipe extérieure introuvable "
                f"dans TikaML: {away_team}"
            ),
        )

    if tika_home == tika_away:

        raise HTTPException(
            status_code=422,
            detail=(
                "Les deux équipes correspondent "
                "à la même équipe TikaML"
            ),
        )

    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    try:

        feature_vector = (
            club_predictor.build_feature_row(
                tika_home,
                tika_away,
                league,
                season,
                match_date,
                week=week,
                odds=None,
            )
        )

    except Exception as exc:

        log.exception(
            "Erreur build_feature_row"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Impossible de construire "
                f"les features: {exc}"
            ),
        )

    # --------------------------------------------------------
    # Current OpenFootball context
    # --------------------------------------------------------

    context = {
        "standings": {},
        "form": {},
        "h2h": {},
    }

    if openfootball_matches:

        context = build_current_context(
            openfootball_matches,
            home_team,
            away_team,
        )

        feature_vector = (
            apply_current_context_to_features(
                feature_vector,
                context,
            )
        )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    predictions = predict_prematch(
        feature_vector,
        [
            "goals",
            "corners",
            "yellows",
        ],
    )

    goals = predictions.get(
        "goals",
        {},
    )

    confidence = (
        calculate_confidence(
            goals,
            context,
        )
    )

    predicted_side = (
        confidence[
            "predicted_side"
        ]
    )

    side_labels = {
        "home": home_team,
        "draw": "Nul",
        "away": away_team,
    }

    best_pick = (
        side_labels.get(
            predicted_side
        )
        if predicted_side
        else None
    )

    return {
        "status": "success",

        "match": {
            "home_team": home_team,
            "away_team": away_team,

            "tika_home_team": tika_home,
            "tika_away_team": tika_away,

            "date": match_date,

            "league": {
                "name": (
                    SUPPORTED_LEAGUES[
                        league
                    ]["name"]
                ),
                "code": league,
            },
        },

        "prediction": {
            "best_pick": best_pick,
            "predicted_side": predicted_side,

            "home_probability": (
                goals.get(
                    "home_win"
                )
            ),

            "draw_probability": (
                goals.get(
                    "draw"
                )
            ),

            "away_probability": (
                goals.get(
                    "away_win"
                )
            ),

            "recommended_score": (
                goals.get(
                    "recommended_score"
                )
            ),

            "expected_goals": {
                "home": goals.get(
                    "expected_home"
                ),
                "away": goals.get(
                    "expected_away"
                ),
                "total": goals.get(
                    "predicted_total"
                ),
            },

            "over_under": goals.get(
                "over_under"
            ),
        },

        "corners": predictions.get(
            "corners"
        ),

        "yellows": predictions.get(
            "yellows"
        ),

        "current_context": context,

        "confidence": confidence,

        "odds": {
            "available": False,
            "provider": None,
            "reason": (
                "OpenFootball ne fournit "
                "pas les cotes bookmakers."
            ),
        },

        "model": {
            "name": (
                "TikaML MatchPredictor"
            ),
            "version": models.version,
            "source": (
                "TikaML historical data "
                "+ OpenFootball current context"
            ),
        },
    }


# ============================================================
# FIND TODAY MATCHES
# ============================================================

def get_today_matches():

    today = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d"
    )

    all_data, errors = (
        load_all_openfootball()
    )

    fixtures = []

    for league_code, data in (
        all_data.items()
    ):

        matches = data.get(
            "matches",
            [],
        )

        for match in matches:

            match_date = (
                parse_match_date(
                    match.get("date")
                )
            )

            if match_date != today:
                continue

            # Match déjà terminé
            if is_completed_match(
                match
            ):
                continue

            home = match.get(
                "team1"
            )

            away = match.get(
                "team2"
            )

            if not home or not away:
                continue

            fixtures.append(
                {
                    "league_code": league_code,
                    "league": SUPPORTED_LEAGUES[
                        league_code
                    ],
                    "match": match,
                }
            )

    return (
        today,
        fixtures,
        all_data,
        errors,
    )


# ============================================================
# TODAY INTERNAL
# ============================================================

def build_today_predictions():

    today, fixtures, all_data, errors = (
        get_today_matches()
    )

    predictions = []
    skipped = []

    season = (
        OPENFOOTBALL_SEASON.replace(
            "-",
            "-",
        )
    )

    # TikaML utilise généralement
    # une saison complète du type 2026-2027
    if re.fullmatch(
        r"\d{4}-\d{2}",
        season,
    ):

        season = (
            season[:4]
            + "-"
            + str(
                int(season[:4])
                + 1
            )
        )

    for fixture in fixtures:

        league_code = fixture[
            "league_code"
        ]

        league_info = fixture[
            "league"
        ]

        match = fixture[
            "match"
        ]

        home = match.get(
            "team1"
        )

        away = match.get(
            "team2"
        )

        match_date = parse_match_date(
            match.get("date")
        )

        kickoff = match.get(
            "time"
        )

        week = get_match_round(
            match
        )

        tika_home = resolve_tika_team(
            home
        )

        tika_away = resolve_tika_team(
            away
        )

        if not tika_home or not tika_away:

            skipped.append(
                {
                    "match": {
                        "home_team": home,
                        "away_team": away,
                    },

                    "league": (
                        league_info["name"]
                    ),

                    "reason": (
                        "Équipe absente "
                        "du dataset TikaML"
                    ),

                    "tika_home_team": (
                        tika_home
                    ),

                    "tika_away_team": (
                        tika_away
                    ),
                }
            )

            continue

        try:

            league_matches = (
                all_data[
                    league_code
                ].get(
                    "matches",
                    [],
                )
            )

            result = gagne_temps_predict(
                home_team=home,
                away_team=away,
                league=league_code,
                season=season,
                match_date=match_date,
                week=week,
                openfootball_matches=(
                    league_matches
                ),
            )

            result["fixture_id"] = (
                f"{league_code}-"
                f"{match_date}-"
                f"{normalize_team_name(home)}-"
                f"{normalize_team_name(away)}"
            )

            result["kickoff"] = kickoff

            result["league"] = {
                "name": (
                    league_info[
                        "name"
                    ]
                ),
                "code": league_code,
                "country": (
                    league_info[
                        "country"
                    ]
                ),
                "openfootball": (
                    league_info[
                        "openfootball"
                    ]
                ),
            }

            predictions.append(
                result
            )

        except HTTPException as exc:

            skipped.append(
                {
                    "match": {
                        "home_team": home,
                        "away_team": away,
                    },

                    "league": (
                        league_info["name"]
                    ),

                    "reason": (
                        exc.detail
                    ),
                }
            )

        except Exception as exc:

            log.exception(
                "Erreur prediction %s - %s",
                home,
                away,
            )

            skipped.append(
                {
                    "match": {
                        "home_team": home,
                        "away_team": away,
                    },

                    "league": (
                        league_info["name"]
                    ),

                    "reason": str(exc),
                }
            )

    predictions.sort(
        key=lambda x: (
            -float(
                x.get(
                    "confidence",
                    {}
                ).get(
                    "score",
                    0,
                )
            ),
            x.get(
                "kickoff"
            )
            or "99:99",
        )
    )

    return {
        "date": today,

        "source": (
            "OpenFootball + TikaML"
        ),

        "model": (
            "TikaML MatchPredictor"
        ),

        "model_version": (
            models.version
        ),

        "season": season,

        "count": len(
            predictions
        ),

        "predictions": predictions,

        "skipped": skipped,

        "source_errors": errors,

        "data_provider": {
            "name": "OpenFootball",
            "api_key_required": False,
            "season": (
                OPENFOOTBALL_SEASON
            ),
        },
    }


# ============================================================
# STANDARD ROUTES
# ============================================================

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

    start = time.time()

    if (
        req.prediction_type
        == "live"
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
        (
            time.time()
            - start
        )
        * 1000,
        1,
    )

    log.info(
        "predict match_id=%s type=%s models=%s elapsed=%sms",
        req.match_id,
        req.prediction_type,
        list(
            predictions.keys()
        ),
        elapsed,
    )

    return PredictionResponse(
        predictions=predictions,
        model_metadata={
            "version": models.version,
            "prediction_type": (
                req.prediction_type
            ),
            "elapsed_ms": elapsed,
        },
    )


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

    start = time.time()

    predictions = predict_prematch(
        req.feature_vector,
        req.models,
        m=backfill_models,
    )

    elapsed = round(
        (
            time.time()
            - start
        )
        * 1000,
        1,
    )

    return PredictionResponse(
        predictions=predictions,
        model_metadata={
            "version": (
                backfill_models.version
            ),
            "prediction_type": (
                "prematch"
            ),
            "elapsed_ms": elapsed,
        },
    )


# ============================================================
# MODEL STATUS
# ============================================================

@app.get(
    "/model-status"
)
async def model_status():

    return {
        "status": "ok",

        "models_loaded": {
            "goals": (
                models.goals
                is not None
            ),

            "corners": (
                models.corners
                is not None
            ),

            "yellows": (
                models.yellows
                is not None
            ),
        },

        "version": models.version,

        "backfill": {
            "loaded": (
                backfill_models.goals
                is not None
            ),
            "version": (
                backfill_models.version
            ),
        },

        "gagne_temps": {
            "loaded": (
                club_predictor
                is not None
            ),

            "historical_matches": (
                len(
                    club_predictor.df
                )
                if (
                    club_predictor
                    is not None
                    and club_predictor.df
                    is not None
                )
                else 0
            ),

            "provider": (
                "OpenFootball"
            ),
        },

        "football_data": {
            "provider": "OpenFootball",
            "api_key_required": False,
            "source": (
                "GitHub raw JSON"
            ),

            "repository": (
                "openfootball/football.json"
            ),

            "current_season": (
                OPENFOOTBALL_SEASON
            ),

            "leagues": [
                {
                    "code": code,
                    "name": info["name"],
                    "file": info["file"],
                }
                for code, info
                in SUPPORTED_LEAGUES.items()
            ],
        },

        "national": {
            "loaded": (
                getattr(
                    national_api.state,
                    "model",
                    None,
                )
                is not None
            ),

            "teams": (
                len(
                    national_api.state.model.attack
                )
                if getattr(
                    national_api.state,
                    "model",
                    None,
                )
                else 0
            ),
        },
    }


# ============================================================
# GAGNE TEMPS - MANUAL PREDICTION
# ============================================================

@app.post(
    "/gagne-temps/predict",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_manual(
    req: GagneTempsRequest,
):

    matches = None

    # Try OpenFootball context automatically
    try:

        data = load_openfootball_league(
            req.league
        )

        matches = data.get(
            "matches",
            [],
        )

    except Exception as exc:

        log.warning(
            "Impossible de charger "
            "OpenFootball pour %s: %s",
            req.league,
            exc,
        )

    return _json_clean(
        gagne_temps_predict(
            home_team=req.home_team,
            away_team=req.away_team,
            league=req.league,
            season=req.season,
            match_date=req.match_date,
            week=req.week,
            openfootball_matches=matches,
        )
    )


# ============================================================
# GAGNE TEMPS - TODAY
# ============================================================

@app.get(
    "/gagne-temps/today",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_today():

    start = time.time()

    result = build_today_predictions()

    result[
        "generated_at"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    result[
        "elapsed_ms"
    ] = round(
        (
            time.time()
            - start
        )
        * 1000,
        1,
    )

    return _json_clean(
        result
    )


# ============================================================
# GAGNE TEMPS - TOP 5
# ============================================================

@app.get(
    "/gagne-temps/top",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_top():

    result = build_today_predictions()

    predictions = result.get(
        "predictions",
        [],
    )

    top = []

    for rank, item in enumerate(
        predictions[:5],
        start=1,
    ):

        prediction = item.get(
            "prediction",
            {}
        )

        confidence = item.get(
            "confidence",
            {}
        )

        top.append(
            {
                "rank": rank,

                "fixture_id": item.get(
                    "fixture_id"
                ),

                "match": {
                    "home_team": (
                        item.get(
                            "match",
                            {}
                        ).get(
                            "home_team"
                        )
                    ),

                    "away_team": (
                        item.get(
                            "match",
                            {}
                        ).get(
                            "away_team"
                        )
                    ),

                    "tika_home_team": (
                        item.get(
                            "match",
                            {}
                        ).get(
                            "tika_home_team"
                        )
                    ),

                    "tika_away_team": (
                        item.get(
                            "match",
                            {}
                        ).get(
                            "tika_away_team"
                        )
                    ),
                },

                "league": item.get(
                    "league"
                ),

                "kickoff": item.get(
                    "kickoff"
                ),

                "best_pick": (
                    prediction.get(
                        "best_pick"
                    )
                ),

                "predicted_side": (
                    prediction.get(
                        "predicted_side"
                    )
                ),

                "probabilities": {
                    "home": (
                        prediction.get(
                            "home_probability"
                        )
                    ),

                    "draw": (
                        prediction.get(
                            "draw_probability"
                        )
                    ),

                    "away": (
                        prediction.get(
                            "away_probability"
                        )
                    ),
                },

                "recommended_score": (
                    prediction.get(
                        "recommended_score"
                    )
                ),

                "confidence": confidence,
            }
        )

    return _json_clean(
        {
            "status": "success",

            "date": result.get(
                "date"
            ),

            "source": (
                "OpenFootball + TikaML"
            ),

            "model": (
                "TikaML MatchPredictor"
            ),

            "version": (
                models.version
            ),

            "count": len(top),

            "top_5": top,

            "skipped": result.get(
                "skipped",
                [],
            ),

            "source_errors": result.get(
                "source_errors",
                {},
            ),
        }
    )


# ============================================================
# GAGNE TEMPS DEBUG
# ============================================================

@app.get(
    "/gagne-temps/debug",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def gagne_temps_debug():

    today, fixtures, all_data, errors = (
        get_today_matches()
    )

    debug_fixtures = []

    for fixture in fixtures:

        match = fixture[
            "match"
        ]

        home = match.get(
            "team1"
        )

        away = match.get(
            "team2"
        )

        debug_fixtures.append(
            {
                "league": fixture[
                    "league"
                ]["name"],

                "home_team": home,

                "away_team": away,

                "tika_home_team": (
                    resolve_tika_team(
                        home
                    )
                ),

                "tika_away_team": (
                    resolve_tika_team(
                        away
                    )
                ),

                "date": parse_match_date(
                    match.get("date")
                ),

                "time": match.get(
                    "time"
                ),

                "round": match.get(
                    "round"
                ),

                "completed": (
                    is_completed_match(
                        match
                    )
                ),
            }
        )

    return _json_clean(
        {
            "date": today,

            "openfootball_season": (
                OPENFOOTBALL_SEASON
            ),

            "fixtures_found": len(
                debug_fixtures
            ),

            "fixtures": (
                debug_fixtures
            ),

            "source_errors": errors,

            "tika_team_count": len(
                _get_tika_team_names()
            ),

            "tika_predictor_loaded": (
                club_predictor
                is not None
            ),
        }
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name": (
            "GAGNE TEMPS"
        ),

        "status": "online",

        "version": "2.0.0",

        "description": (
            "Football prediction API "
            "powered by TikaML + OpenFootball"
        ),

        "endpoints": {
            "health": "/model-status",
            "prediction": "/predict",
            "backfill": "/backfill",
            "gagne_temps": (
                "/gagne-temps/predict"
            ),
            "today": (
                "/gagne-temps/today"
            ),
            "top": (
                "/gagne-temps/top"
            ),
            "debug": (
                "/gagne-temps/debug"
            ),
        },

        "football_data": {
            "provider": "OpenFootball",
            "season": (
                OPENFOOTBALL_SEASON
            ),
            "leagues": [
                info["name"]
                for info
                in SUPPORTED_LEAGUES.values()
            ],
        },
}
