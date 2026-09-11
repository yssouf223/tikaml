"""
GAGNE TEMPS + TikaML Prediction API
===================================

Football prediction API based on:

- TikaML MatchPredictor
- LightGBM + Poisson
- OpenFootball
- Historical TikaML features

IMPORTANT
---------
Cette version corrige définitivement :

    'LGBMPoissonModel' object has no attribute 'predict'

Le serveur utilise :

    MatchPredictor.predict()

et possède également un moteur de secours interne qui utilise :

    model.predict_lambdas()
    model.predict_score_matrix()

Il n'utilise JAMAIS :

    model.predict()

Public:
    GET  /
    GET  /health
    GET  /gagne-temps/health
    GET  /gagne-temps/leagues
    GET  /gagne-temps/today
    GET  /gagne-temps/top
    POST /gagne-temps/predict

Protected:
    POST /predict
    POST /predict/live
    POST /backfill
    GET  /model-status
    GET  /debug/openfootball/{league_code}

Authentication:
    X-API-Key
    TIKA_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Security,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


# ============================================================
# TIKAML IMPORTS
# ============================================================

try:
    from src.inference import MatchPredictor
except Exception as exc:
    MatchPredictor = None
    print(f"WARNING: MatchPredictor import failed: {exc}")


try:
    from src.lgbm_poisson import (
        LGBMPoissonModel,
        FEATURE_COLS,
        CORNER_FEATURE_COLS,
        YELLOW_FEATURE_COLS,
    )
except Exception as exc:
    LGBMPoissonModel = None
    FEATURE_COLS = []
    CORNER_FEATURE_COLS = []
    YELLOW_FEATURE_COLS = []
    print(f"WARNING: TikaML model import failed: {exc}")


try:
    from src.live_predictor import LivePredictor
except Exception as exc:
    LivePredictor = None
    print(f"WARNING: LivePredictor import failed: {exc}")


try:
    from src import national_api
except Exception as exc:
    national_api = None
    print(f"WARNING: national_api import failed: {exc}")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("gagne-temps")


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "GAGNE TEMPS"
APP_VERSION = "2.2.0"

MAX_GOALS = 7

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = MODEL_DIR / "corners"
YELLOW_MODEL_DIR = MODEL_DIR / "yellows"

BACKFILL_DIR = MODEL_DIR / "backfill_20260131"

FEATURES_FILE = Path(
    "data/opta/processed/features.csv"
)

OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)

OPENFOOTBALL_SEASON = "2026-27"

OPENFOOTBALL_TIMEOUT = 20

CACHE_TTL_SECONDS = 300


# ============================================================
# LEAGUES
# ============================================================

LEAGUES: dict[str, dict[str, str]] = {
    "EPL": {
        "name": "Premier League",
        "country": "England",
        "file": "en.1.json",
    },
    "LL": {
        "name": "La Liga",
        "country": "Spain",
        "file": "es.1.json",
    },
    "SEA": {
        "name": "Serie A",
        "country": "Italy",
        "file": "it.1.json",
    },
    "BUN": {
        "name": "Bundesliga",
        "country": "Germany",
        "file": "de.1.json",
    },
    "LI1": {
        "name": "Ligue 1",
        "country": "France",
        "file": "fr.1.json",
    },
}


# ============================================================
# API KEY
# ============================================================

API_KEY = os.environ.get(
    "TIKA_API_KEY",
    "",
).strip()

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)

    log.warning(
        "TIKA_API_KEY is not configured. "
        "A temporary API key was generated."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str | None = Security(api_key_header),
):
    if not key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
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
# GLOBAL OBJECTS
# ============================================================

club_predictor: Any = None
backfill_predictor: Any = None
live_predictor: Any = None

MODEL_VERSION = "unknown"


# ============================================================
# OPENFOOTBALL CACHE
# ============================================================

_openfootball_cache: dict[
    str,
    dict[str, Any],
] = {}


def cache_get(
    key: str,
) -> Any | None:

    item = _openfootball_cache.get(key)

    if item is None:
        return None

    if (
        time.time() - item["timestamp"]
        > CACHE_TTL_SECONDS
    ):
        _openfootball_cache.pop(
            key,
            None,
        )
        return None

    return item["data"]


def cache_set(
    key: str,
    data: Any,
) -> None:

    _openfootball_cache[key] = {
        "timestamp": time.time(),
        "data": data,
    }


# ============================================================
# TEAM ALIASES
# ============================================================

TEAM_ALIASES: dict[str, list[str]] = {
    "1. FC Union Berlin": [
        "1. FC Union Berlin",
        "Union Berlin",
        "1. FC Union",
    ],
    "FC Schalke 04": [
        "FC Schalke 04",
        "Schalke 04",
        "Schalke",
    ],
    "Stade Rennais FC 1901": [
        "Stade Rennais FC 1901",
        "Rennes",
        "Stade Rennais",
    ],
    "Olympique de Marseille": [
        "Olympique de Marseille",
        "Olympique Marseille",
        "Marseille",
    ],
    "Sevilla FC": [
        "Sevilla FC",
        "Sevilla",
    ],
    "Valencia CF": [
        "Valencia CF",
        "Valencia",
    ],
    "Venezia FC": [
        "Venezia FC",
        "Venezia",
    ],
    "ACF Fiorentina": [
        "ACF Fiorentina",
        "Fiorentina",
    ],
}


DIRECT_TIKA_ALIASES = {
    "1 fc union berlin": "Union Berlin",
    "fc schalke 04": "Schalke 04",
    "stade rennais fc 1901": "Rennes",
    "olympique de marseille": "Olympique Marseille",
    "sevilla fc": "Sevilla",
    "valencia cf": "Valencia",
    "venezia fc": "Venezia",
    "acf fiorentina": "Fiorentina",
}


def normalize_team_name(
    name: str,
) -> str:

    if not name:
        return ""

    value = str(name).strip().lower()

    value = (
        value
        .replace("’", "'")
        .replace("-", " ")
        .replace("_", " ")
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def resolve_tika_team(
    openfootball_name: str,
    available_teams: list[str] | None = None,
) -> str:

    if not openfootball_name:
        raise ValueError(
            "Team name is empty"
        )

    normalized = normalize_team_name(
        openfootball_name
    )

    # Direct mappings
    direct = DIRECT_TIKA_ALIASES.get(
        normalized
    )

    if direct:
        if available_teams:
            direct_norm = normalize_team_name(
                direct
            )

            for team in available_teams:
                if (
                    normalize_team_name(team)
                    == direct_norm
                ):
                    return team

        return direct

    # Candidate aliases
    candidates = [
        openfootball_name
    ]

    candidates.extend(
        TEAM_ALIASES.get(
            openfootball_name,
            [],
        )
    )

    normalized_candidates = {
        normalize_team_name(x)
        for x in candidates
        if x
    }

    # Exact
    if available_teams:

        for team in available_teams:

            team_norm = normalize_team_name(
                team
            )

            if team_norm in normalized_candidates:
                return team

        # Conservative partial matching
        for team in available_teams:

            team_norm = normalize_team_name(
                team
            )

            for candidate in normalized_candidates:

                if (
                    len(candidate) >= 5
                    and (
                        candidate in team_norm
                        or team_norm in candidate
                    )
                ):
                    return team

    return openfootball_name


# ============================================================
# OPENFOOTBALL
# ============================================================

def openfootball_url(
    league_code: str,
    season: str = OPENFOOTBALL_SEASON,
) -> str:

    config = LEAGUES.get(
        league_code
    )

    if not config:
        raise ValueError(
            f"Unknown league: {league_code}"
        )

    return (
        f"{OPENFOOTBALL_BASE}/"
        f"{season}/"
        f"{config['file']}"
    )


def fetch_openfootball_league(
    league_code: str,
    season: str = OPENFOOTBALL_SEASON,
) -> list[dict[str, Any]]:

    cache_key = (
        f"{season}:{league_code}"
    )

    cached = cache_get(
        cache_key
    )

    if cached is not None:
        return cached

    url = openfootball_url(
        league_code,
        season,
    )

    response = requests.get(
        url,
        timeout=OPENFOOTBALL_TIMEOUT,
        headers={
            "User-Agent": "GAGNE-TEMPS/2.2",
        },
    )

    response.raise_for_status()

    data = response.json()

    matches: list[
        dict[str, Any]
    ] = []

    for round_item in data.get(
        "rounds",
        [],
    ):

        round_name = (
            round_item.get("name")
            or round_item.get("round")
        )

        for match in round_item.get(
            "matches",
            [],
        ):

            item = dict(match)

            item["_league"] = league_code
            item["_season"] = season
            item["_round"] = round_name

            matches.append(item)

    cache_set(
        cache_key,
        matches,
    )

    return matches


def fetch_all_openfootball(
    season: str = OPENFOOTBALL_SEASON,
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
]:

    all_matches: list[
        dict[str, Any]
    ] = []

    errors: dict[str, str] = {}

    for league_code in LEAGUES:

        try:

            matches = fetch_openfootball_league(
                league_code,
                season,
            )

            all_matches.extend(
                matches
            )

        except Exception as exc:

            log.exception(
                "OpenFootball failed for %s",
                league_code,
            )

            errors[
                league_code
            ] = str(exc)

    return (
        all_matches,
        errors,
    )


# ============================================================
# DATE / SEASON
# ============================================================

def parse_date(
    value: Any,
) -> pd.Timestamp | None:

    if value is None:
        return None

    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def normalize_date(
    value: Any,
) -> str | None:

    parsed = parse_date(value)

    if parsed is None:
        return None

    return parsed.strftime(
        "%Y-%m-%d"
    )


def normalize_season(
    season: str | None,
) -> str:

    if not season:
        return "2026-2027"

    value = str(
        season
    ).strip()

    match = re.fullmatch(
        r"(\d{4})-(\d{2}|\d{4})",
        value,
    )

    if not match:
        return value

    first = match.group(1)
    second = match.group(2)

    if len(second) == 2:
        second = (
            first[:2] + second
        )

    return f"{first}-{second}"


def round_to_week(
    round_name: str | None,
) -> int | None:

    if not round_name:
        return None

    match = re.search(
        r"(\d+)",
        str(round_name),
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def match_has_final_score(
    match: dict[str, Any],
) -> bool:

    score = match.get(
        "score"
    )

    if not isinstance(
        score,
        dict,
    ):
        return False

    ft = score.get(
        "ft"
    )

    if ft is None:
        return False

    if isinstance(
        ft,
        (list, tuple),
    ):
        return len(ft) >= 2

    return True


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): clean_json(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            clean_json(v)
            for v in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return clean_json(
            value.tolist()
        )

    if isinstance(
        value,
        np.generic,
    ):
        return clean_json(
            value.item()
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        pd.Series,
    ):
        return clean_json(
            value.to_dict()
        )

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return clean_json(
            value.to_dict(
                orient="records"
            )
        )

    if isinstance(
        value,
        bool,
    ):
        return bool(value)

    if isinstance(
        value,
        (int, np.integer),
    ):
        return int(value)

    if isinstance(
        value,
        (float, np.floating),
    ):
        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    try:

        missing = pd.isna(
            value
        )

        if isinstance(
            missing,
            (
                bool,
                np.bool_,
            ),
        ):

            if bool(missing):
                return None

    except Exception:
        pass

    return value


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    probs: list[float] | np.ndarray,
) -> tuple[float, str]:

    values = np.asarray(
        probs,
        dtype=float,
    )

    if values.size != 3:
        return (
            0.0,
            "FAIBLE",
        )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = values.sum()

    if total <= 0:
        return (
            0.0,
            "FAIBLE",
        )

    values /= total

    ordered = np.sort(
        values
    )[::-1]

    best = float(
        ordered[0]
    )

    second = float(
        ordered[1]
    )

    margin = max(
        0.0,
        best - second,
    )

    confidence = (
        0.55 * best
        + 0.45
        * min(
            1.0,
            margin * 2.5,
        )
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    percentage = (
        confidence * 100
    )

    if percentage >= 75:
        label = "FORTE"
    elif percentage >= 60:
        label = "MOYENNE"
    else:
        label = "FAIBLE"

    return (
        round(
            percentage,
            1,
        ),
        label,
    )


# ============================================================
# SCORE GROUPS
# ============================================================

def build_score_groups(
    matrix: np.ndarray,
) -> list[dict[str, Any]]:

    groups = {
        "home_win": [],
        "draw": [],
        "away_win": [],
    }

    rows, cols = matrix.shape

    for i in range(rows):

        for j in range(cols):

            probability = float(
                matrix[i, j]
            )

            if i > j:
                group = "home_win"
            elif i == j:
                group = "draw"
            else:
                group = "away_win"

            groups[group].append(
                (
                    i,
                    j,
                    probability,
                )
            )

    result = []

    for group_name, scores in groups.items():

        scores.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        total = sum(
            x[2]
            for x in scores
        )

        result.append(
            {
                "type": group_name,
                "prob": total,
                "top_scores": scores[:5],
            }
        )

    return result


# ============================================================
# TIKAML RESULT CONVERSION
# ============================================================

def convert_tikaml_prediction(
    result: dict[str, Any],
    home_team: str,
    away_team: str,
    league: str,
    kickoff: str | None = None,
) -> dict[str, Any]:

    probs = result.get(
        "probs_1x2",
        [0.0, 0.0, 0.0],
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    if probs.size != 3:
        probs = np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        )

    probs = np.nan_to_num(
        probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = probs.sum()

    if total > 0:
        probs /= total

    p_home = float(
        probs[0]
    )

    p_draw = float(
        probs[1]
    )

    p_away = float(
        probs[2]
    )

    outcome_index = int(
        np.argmax(probs)
    )

    sides = [
        "home",
        "draw",
        "away",
    ]

    predicted_side = sides[
        outcome_index
    ]

    confidence, confidence_label = (
        calculate_confidence(
            probs
        )
    )

    if predicted_side == "home":
        best_pick = home_team
    elif predicted_side == "away":
        best_pick = away_team
    else:
        best_pick = "Match nul"

    recommended = result.get(
        "recommended_score",
        {},
    )

    score_label = None
    score_probability = 0.0

    if isinstance(
        recommended,
        dict,
    ):

        score_label = recommended.get(
            "label"
        )

        if not score_label:

            hg = recommended.get(
                "home_goals"
            )

            ag = recommended.get(
                "away_goals"
            )

            if (
                hg is not None
                and ag is not None
            ):

                score_label = (
                    f"{int(hg)}-{int(ag)}"
                )

        score_probability = float(
            recommended.get(
                "prob",
                0.0,
            )
            or 0.0
        )

    if not score_label:
        score_label = "N/A"

    ordered = np.sort(
        probs
    )[::-1]

    margin = float(
        ordered[0] - ordered[1]
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "kickoff": kickoff,
        "best_pick": best_pick,
        "predicted_side": predicted_side,

        "probabilities": {
            "home": round(
                p_home,
                4,
            ),
            "draw": round(
                p_draw,
                4,
            ),
            "away": round(
                p_away,
                4,
            ),
        },

        "recommended_score": {
            "label": score_label,
            "prob": round(
                score_probability,
                4,
            ),
        },

        "confidence": confidence,
        "confidence_label": confidence_label,

        "margin": round(
            margin,
            4,
        ),

        "lambda_home": result.get(
            "lambda_home"
        ),

        "lambda_away": result.get(
            "lambda_away"
        ),

        "top_scores": result.get(
            "top_scores",
            [],
        ),

        "score_groups": result.get(
            "score_groups",
            [],
        ),

        "goals_over_under": result.get(
            "goals_over_under",
            {},
        ),

        "corners": result.get(
            "corners"
        ),

        "yellows": result.get(
            "yellows"
        ),
    }


# ============================================================
# CRITICAL FIX
# ============================================================

def direct_tikaml_prediction(
    predictor: Any,
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    match_date: str,
    week: int | None = None,
    max_goals: int = MAX_GOALS,
    odds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Fallback TikaML prediction engine.

    IMPORTANT:
    This function NEVER calls:

        predictor.model.predict()

    It uses only:

        predictor.build_feature_row()
        predictor.model.predict_lambdas()
        predictor.model.predict_score_matrix()

    This is compatible with the current LGBMPoissonModel.
    """

    if predictor is None:
        raise RuntimeError(
            "TikaML predictor is not loaded"
        )

    model = getattr(
        predictor,
        "model",
        None,
    )

    if model is None:
        raise RuntimeError(
            "TikaML goals model is not loaded"
        )

    if not hasattr(
        predictor,
        "build_feature_row",
    ):
        raise RuntimeError(
            "MatchPredictor.build_feature_row() is unavailable"
        )

    if not hasattr(
        model,
        "predict_lambdas",
    ):
        raise RuntimeError(
            "LGBMPoissonModel.predict_lambdas() is unavailable"
        )

    if not hasattr(
        model,
        "predict_score_matrix",
    ):
        raise RuntimeError(
            "LGBMPoissonModel.predict_score_matrix() is unavailable"
        )

    if isinstance(
        match_date,
        str,
    ):
        match_date_pd = pd.Timestamp(
            match_date
        )
    else:
        match_date_pd = match_date

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = predictor.build_feature_row(
        home_team,
        away_team,
        league,
        normalize_season(
            season
        ),
        match_date_pd,
        week,
        odds=odds,
    )

    feat_df = pd.DataFrame(
        [features]
    )

    # --------------------------------------------------------
    # CRITICAL:
    # NEVER use model.predict()
    # --------------------------------------------------------

    lambda_home, lambda_away = (
        model.predict_lambdas(
            feat_df
        )
    )

    lambda_home = float(
        lambda_home[0]
    )

    lambda_away = float(
        lambda_away[0]
    )

    # --------------------------------------------------------
    # Score matrix
    # --------------------------------------------------------

    matrix = model.predict_score_matrix(
        lambda_home,
        lambda_away,
        max_goals,
    )

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.sum() > 0:
        matrix = (
            matrix / matrix.sum()
        )

    # --------------------------------------------------------
    # 1X2
    # --------------------------------------------------------

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

    probs = np.array(
        [
            p_home,
            p_draw,
            p_away,
        ],
        dtype=float,
    )

    if probs.sum() > 0:
        probs /= probs.sum()

    outcome_index = int(
        np.argmax(probs)
    )

    # --------------------------------------------------------
    # Top scores
    # --------------------------------------------------------

    flat = matrix.flatten()

    top_indexes = (
        flat.argsort()[::-1][:5]
    )

    top_scores = []

    for index in top_indexes:

        home_goals, away_goals = (
            divmod(
                int(index),
                max_goals,
            )
        )

        top_scores.append(
            (
                int(home_goals),
                int(away_goals),
                float(
                    flat[index]
                ),
            )
        )

    # --------------------------------------------------------
    # Score groups
    # --------------------------------------------------------

    score_groups = build_score_groups(
        matrix
    )

    group_map = {
        0: "home_win",
        1: "draw",
        2: "away_win",
    }

    predicted_group = group_map[
        outcome_index
    ]

    selected_group = next(
        (
            group
            for group in score_groups
            if group["type"]
            == predicted_group
        ),
        None,
    )

    if (
        selected_group
        and selected_group["top_scores"]
    ):

        recommended_tuple = (
            selected_group[
                "top_scores"
            ][0]
        )

        recommended_score = {
            "home_goals": int(
                recommended_tuple[0]
            ),
            "away_goals": int(
                recommended_tuple[1]
            ),
            "prob": float(
                recommended_tuple[2]
            ),
            "label": (
                f"{int(recommended_tuple[0])}-"
                f"{int(recommended_tuple[1])}"
            ),
        }

    else:

        recommended_score = {
            "home_goals": int(
                top_scores[0][0]
            )
            if top_scores
            else 0,
            "away_goals": int(
                top_scores[0][1]
            )
            if top_scores
            else 0,
            "prob": float(
                top_scores[0][2]
            )
            if top_scores
            else 0.0,
            "label": (
                f"{int(top_scores[0][0])}-"
                f"{int(top_scores[0][1])}"
            )
            if top_scores
            else "0-0",
        }

    # --------------------------------------------------------
    # Goals O/U
    # --------------------------------------------------------

    goals_over_under = {}

    for line in [
        0.5,
        1.5,
        2.5,
        3.5,
        4.5,
    ]:

        over_probability = 0.0

        for i in range(
            matrix.shape[0]
        ):

            for j in range(
                matrix.shape[1]
            ):

                if i + j > line:
                    over_probability += float(
                        matrix[i, j]
                    )

        goals_over_under[
            str(line)
        ] = {
            "over": round(
                over_probability,
                4,
            ),
            "under": round(
                1.0 - over_probability,
                4,
            ),
        }

    # --------------------------------------------------------
    # Corners
    # --------------------------------------------------------

    corners = None

    corner_model = getattr(
        predictor,
        "corner_model",
        None,
    )

    if corner_model is not None:

        try:

            if hasattr(
                corner_model,
                "predict_lambdas",
            ):

                corner_home, corner_away = (
                    corner_model.predict_lambdas(
                        feat_df
                    )
                )

                ch = float(
                    corner_home[0]
                )

                ca = float(
                    corner_away[0]
                )

                corners = {
                    "lambda_home": ch,
                    "lambda_away": ca,
                }

        except Exception as exc:

            log.warning(
                "Corner prediction unavailable: %s",
                exc,
            )

    # --------------------------------------------------------
    # Yellow cards
    # --------------------------------------------------------

    yellows = None

    yellow_model = getattr(
        predictor,
        "yellow_model",
        None,
    )

    if yellow_model is not None:

        try:

            if hasattr(
                yellow_model,
                "predict_lambdas",
            ):

                yellow_home, yellow_away = (
                    yellow_model.predict_lambdas(
                        feat_df
                    )
                )

                yh = float(
                    yellow_home[0]
                )

                ya = float(
                    yellow_away[0]
                )

                yellows = {
                    "lambda_home": yh,
                    "lambda_away": ya,
                }

        except Exception as exc:

            log.warning(
                "Yellow prediction unavailable: %s",
                exc,
            )

    # --------------------------------------------------------
    # Return TikaML-compatible structure
    # --------------------------------------------------------

    return {
        "score_matrix": matrix,
        "probs_1x2": probs,

        "predicted_outcome": [
            "home_win",
            "draw",
            "away_win",
        ][outcome_index],

        "recommended_score": (
            recommended_score
        ),

        "lambda_home": lambda_home,
        "lambda_away": lambda_away,

        "top_scores": top_scores,

        "score_groups": score_groups,

        "goals_over_under": (
            goals_over_under
        ),

        "corners": corners,
        "yellows": yellows,

        "features": features,
    }


# ============================================================
# SAFE TIKAML PREDICTION
# ============================================================

def safe_tikaml_predict(
    predictor: Any,
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    match_date: str,
    week: int | None = None,
    max_goals: int = MAX_GOALS,
    odds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Utilise MatchPredictor.predict() normalement.

    Si la version installée de MatchPredictor contient
    encore l'ancien appel model.predict(), le serveur
    bascule automatiquement vers direct_tikaml_prediction().
    """

    if predictor is None:
        raise RuntimeError(
            "TikaML MatchPredictor is not loaded"
        )

    # --------------------------------------------------------
    # Première tentative : API officielle MatchPredictor
    # --------------------------------------------------------

    try:

        if not hasattr(
            predictor,
            "predict",
        ):
            raise AttributeError(
                "MatchPredictor.predict() unavailable"
            )

        log.info(
            "Using MatchPredictor.predict()"
        )

        return predictor.predict(
            home_team=home_team,
            away_team=away_team,
            league=league,
            season=normalize_season(
                season
            ),
            match_date=match_date,
            week=week,
            max_goals=max_goals,
            odds=odds,
        )

    except AttributeError as exc:

        error_text = str(exc)

        # ----------------------------------------------------
        # CRITICAL FALLBACK
        # ----------------------------------------------------

        if (
            "LGBMPoissonModel"
            in error_text
            and ".predict"
            in error_text
        ):

            log.warning(
                "Detected incompatible TikaML .predict() call. "
                "Switching to direct predict_lambdas() fallback."
            )

            return direct_tikaml_prediction(
                predictor=predictor,
                home_team=home_team,
                away_team=away_team,
                league=league,
                season=season,
                match_date=match_date,
                week=week,
                max_goals=max_goals,
                odds=odds,
            )

        raise

    except Exception as exc:

        error_text = str(exc)

        # ----------------------------------------------------
        # Some older TikaML versions raise the AttributeError
        # wrapped inside another exception.
        # ----------------------------------------------------

        if (
            "LGBMPoissonModel"
            in error_text
            and "predict"
            in error_text
        ):

            log.warning(
                "TikaML compatibility issue detected: %s",
                error_text,
            )

            log.warning(
                "Using direct predict_lambdas() fallback."
            )

            return direct_tikaml_prediction(
                predictor=predictor,
                home_team=home_team,
                away_team=away_team,
                league=league,
                season=season,
                match_date=match_date,
                week=week,
                max_goals=max_goals,
                odds=odds,
            )

        raise


# ============================================================
# GAGNE TEMPS MATCH PREDICTION
# ============================================================

def predict_gagne_temps_match(
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    match_date: str,
    week: int | None = None,
    kickoff: str | None = None,
) -> dict[str, Any]:

    global club_predictor

    if club_predictor is None:

        raise RuntimeError(
            "TikaML MatchPredictor is not loaded"
        )

    # --------------------------------------------------------
    # Get available TikaML teams
    # --------------------------------------------------------

    available_teams: list[str] = []

    try:

        df = getattr(
            club_predictor,
            "df",
            None,
        )

        if df is not None:

            for column in [
                "home_team",
                "away_team",
            ]:

                if column in df.columns:

                    available_teams.extend(
                        df[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

            available_teams = list(
                dict.fromkeys(
                    available_teams
                )
            )

    except Exception as exc:

        log.warning(
            "Could not read TikaML team list: %s",
            exc,
        )

    # --------------------------------------------------------
    # Resolve names
    # --------------------------------------------------------

    home_tika = resolve_tika_team(
        home_team,
        available_teams,
    )

    away_tika = resolve_tika_team(
        away_team,
        available_teams,
    )

    normalized_season = (
        normalize_season(
            season
        )
    )

    log.info(
        "GAGNE TEMPS prediction:"
    )

    log.info(
        "  Home: %s -> %s",
        home_team,
        home_tika,
    )

    log.info(
        "  Away: %s -> %s",
        away_team,
        away_tika,
    )

    log.info(
        "  League: %s",
        league,
    )

    log.info(
        "  Season: %s",
        normalized_season,
    )

    log.info(
        "  Date: %s",
        match_date,
    )

    # --------------------------------------------------------
    # SAFE PREDICTION
    # --------------------------------------------------------

    result = safe_tikaml_predict(
        predictor=club_predictor,
        home_team=home_tika,
        away_team=away_tika,
        league=league,
        season=normalized_season,
        match_date=match_date,
        week=week,
        max_goals=MAX_GOALS,
        odds=None,
    )

    prediction = convert_tikaml_prediction(
        result=result,
        home_team=home_team,
        away_team=away_team,
        league=league,
        kickoff=kickoff,
    )

    prediction[
        "tika_home_team"
    ] = home_tika

    prediction[
        "tika_away_team"
    ] = away_tika

    return clean_json(
        prediction
    )


# ============================================================
# GET PREDICTIONS FOR DATE
# ============================================================

def get_predictions_for_date(
    target_date: str,
    top_only: bool = False,
) -> dict[str, Any]:

    matches, source_errors = (
        fetch_all_openfootball(
            OPENFOOTBALL_SEASON
        )
    )

    predictions: list[
        dict[str, Any]
    ] = []

    skipped: list[
        dict[str, Any]
    ] = []

    for match in matches:

        match_date = normalize_date(
            match.get("date")
        )

        if match_date != target_date:
            continue

        # ----------------------------------------------------
        # Completed match
        # ----------------------------------------------------

        if match_has_final_score(
            match
        ):

            skipped.append(
                {
                    "league": match.get(
                        "_league"
                    ),
                    "home_team": match.get(
                        "team1"
                    ),
                    "away_team": match.get(
                        "team2"
                    ),
                    "reason": "match_completed",
                }
            )

            continue

        home_team = match.get(
            "team1"
        )

        away_team = match.get(
            "team2"
        )

        league_code = match.get(
            "_league"
        )

        round_name = match.get(
            "_round"
        )

        if not home_team or not away_team:

            skipped.append(
                {
                    "league": league_code,
                    "home_team": home_team,
                    "away_team": away_team,
                    "reason": "missing_team",
                }
            )

            continue

        if league_code not in LEAGUES:

            skipped.append(
                {
                    "league": league_code,
                    "home_team": home_team,
                    "away_team": away_team,
                    "reason": "unsupported_league",
                }
            )

            continue

        week = round_to_week(
            round_name
        )

        try:

            prediction = (
                predict_gagne_temps_match(
                    home_team=home_team,
                    away_team=away_team,
                    league=league_code,
                    season=OPENFOOTBALL_SEASON,
                    match_date=match_date,
                    week=week,
                    kickoff=match.get(
                        "time"
                    ),
                )
            )

            league_config = LEAGUES[
                league_code
            ]

            prediction[
                "league_code"
            ] = league_code

            prediction[
                "league_name"
            ] = league_config[
                "name"
            ]

            prediction[
                "country"
            ] = league_config[
                "country"
            ]

            prediction[
                "round"
            ] = round_name

            prediction[
                "date"
            ] = match_date

            prediction[
                "source"
            ] = "OpenFootball + TikaML"

            prediction[
                "model"
            ] = "TikaML MatchPredictor"

            prediction[
                "version"
            ] = MODEL_VERSION

            predictions.append(
                prediction
            )

        except Exception as exc:

            log.exception(
                "Prediction failed: %s vs %s",
                home_team,
                away_team,
            )

            skipped.append(
                {
                    "league": league_code,
                    "home_team": home_team,
                    "away_team": away_team,
                    "reason": str(exc),
                }
            )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    predictions.sort(
        key=lambda item: (
            float(
                item.get(
                    "confidence",
                    0.0,
                )
                or 0.0
            ),
            float(
                item.get(
                    "margin",
                    0.0,
                )
                or 0.0
            ),
        ),
        reverse=True,
    )

    top_5 = predictions[:5]

    response = {
        "status": "success",
        "source": "OpenFootball + TikaML",
        "model": "TikaML MatchPredictor",
        "version": MODEL_VERSION,
        "date": target_date,
        "count": (
            len(top_5)
            if top_only
            else len(predictions)
        ),
        "skipped": skipped,
        "source_errors": source_errors,
    }

    if top_only:
        response["top_5"] = top_5
    else:
        response["predictions"] = predictions
        response["top_5"] = top_5

    return clean_json(
        response
    )


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(
    BaseModel
):

    home_team: str = Field(
        ...,
        min_length=1,
    )

    away_team: str = Field(
        ...,
        min_length=1,
    )

    league: str = Field(
        ...,
        min_length=1,
    )

    season: str = (
        OPENFOOTBALL_SEASON
    )

    match_date: str

    week: int | None = None

    odds: dict[
        str,
        float,
    ] | None = None


class GagneTempsPredictRequest(
    BaseModel
):

    home_team: str = Field(
        ...,
        min_length=1,
    )

    away_team: str = Field(
        ...,
        min_length=1,
    )

    league: str = Field(
        ...,
        min_length=1,
    )

    season: str = (
        OPENFOOTBALL_SEASON
    )

    match_date: str

    week: int | None = None

    kickoff: str | None = None


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global club_predictor
    global backfill_predictor
    global live_predictor
    global MODEL_VERSION

    log.info(
        "=========================================="
    )

    log.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    log.info(
        "OpenFootball season: %s",
        OPENFOOTBALL_SEASON,
    )

    # ========================================================
    # MAIN MATCH PREDICTOR
    # ========================================================

    if MatchPredictor is None:

        log.error(
            "MatchPredictor import failed"
        )

    else:

        try:

            club_predictor = MatchPredictor()

            club_predictor.load_model()

            log.info(
                "TikaML MatchPredictor loaded successfully"
            )

            model = getattr(
                club_predictor,
                "model",
                None,
            )

            if model is None:

                log.error(
                    "TikaML goals model is missing"
                )

            else:

                log.info(
                    "TikaML goals model loaded"
                )

                log.info(
                    "Model class: %s",
                    type(model).__name__,
                )

                log.info(
                    "predict_lambdas available: %s",
                    hasattr(
                        model,
                        "predict_lambdas",
                    ),
                )

                log.info(
                    "predict_score_matrix available: %s",
                    hasattr(
                        model,
                        "predict_score_matrix",
                    ),
                )

                log.info(
                    "generic model.predict available: %s",
                    hasattr(
                        model,
                        "predict",
                    ),
                )

        except Exception as exc:

            club_predictor = None

            log.exception(
                "Could not load TikaML MatchPredictor: %s",
                exc,
            )

    # ========================================================
    # MODEL VERSION
    # ========================================================

    meta_file = (
        MODEL_DIR / "meta.json"
    )

    if meta_file.exists():

        try:

            with meta_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                meta = json.load(
                    file
                )

            feature_count = len(
                meta.get(
                    "feature_cols",
                    [],
                )
            )

            if feature_count:

                MODEL_VERSION = (
                    f"lgbm-poisson-"
                    f"{feature_count}f"
                )

            else:

                MODEL_VERSION = str(
                    meta.get(
                        "version",
                        "unknown",
                    )
                )

        except Exception as exc:

            log.warning(
                "Could not read meta.json: %s",
                exc,
            )

    if (
        MODEL_VERSION == "unknown"
        and club_predictor is not None
    ):

        MODEL_VERSION = (
            "lgbm-poisson"
        )

    log.info(
        "Model version: %s",
        MODEL_VERSION,
    )

    # ========================================================
    # LIVE PREDICTOR
    # ========================================================

    if LivePredictor is not None:

        try:

            live_predictor = LivePredictor()

            if hasattr(
                live_predictor,
                "load_model",
            ):

                live_predictor.load_model()

            log.info(
                "LivePredictor loaded"
            )

        except Exception as exc:

            live_predictor = None

            log.warning(
                "LivePredictor unavailable: %s",
                exc,
            )

    # ========================================================
    # BACKFILL
    # ========================================================

    if (
        MatchPredictor is not None
        and FEATURES_FILE.exists()
    ):

        try:

            backfill_predictor = (
                MatchPredictor(
                    features_path=str(
                        FEATURES_FILE
                    )
                )
            )

            backfill_goals = (
                BACKFILL_DIR / "goals"
            )

            backfill_corners = (
                BACKFILL_DIR / "corners"
            )

            backfill_yellows = (
                BACKFILL_DIR / "yellows"
            )

            if (
                LGBMPoissonModel is not None
                and backfill_goals.exists()
            ):

                backfill_predictor.model = (
                    LGBMPoissonModel.load(
                        str(
                            backfill_goals
                        )
                    )
                )

            if (
                LGBMPoissonModel is not None
                and backfill_corners.exists()
            ):

                backfill_predictor.corner_model = (
                    LGBMPoissonModel.load(
                        str(
                            backfill_corners
                        )
                    )
                )

            if (
                LGBMPoissonModel is not None
                and backfill_yellows.exists()
            ):

                backfill_predictor.yellow_model = (
                    LGBMPoissonModel.load(
                        str(
                            backfill_yellows
                        )
                    )
                )

            log.info(
                "Backfill predictor initialized"
            )

        except Exception as exc:

            backfill_predictor = None

            log.warning(
                "Backfill predictor unavailable: %s",
                exc,
            )

    # ========================================================
    # NATIONAL API
    # ========================================================

    if national_api is not None:

        log.info(
            "National API module detected"
        )

    log.info(
        "=========================================="
    )

    log.info(
        "GAGNE TEMPS startup completed"
    )

    log.info(
        "Predictor loaded: %s",
        club_predictor is not None,
    )

    log.info(
        "=========================================="
    )

    yield

    log.info(
        "GAGNE TEMPS shutting down"
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "Football prediction API powered "
        "by TikaML + OpenFootball"
    ),
    version=APP_VERSION,
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
# NATIONAL API
# ============================================================

if national_api is not None:

    try:

        app.include_router(
            national_api.router,
            dependencies=[
                Depends(
                    verify_api_key
                )
            ],
        )

    except Exception as exc:

        log.warning(
            "National router unavailable: %s",
            exc,
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "source": "OpenFootball + TikaML",
        "model": "TikaML MatchPredictor",
        "model_version": MODEL_VERSION,
        "predictor_loaded": (
            club_predictor is not None
        ),
        "season": OPENFOOTBALL_SEASON,
        "leagues": list(
            LEAGUES.keys()
        ),
        "public_endpoints": [
            "/",
            "/health",
            "/gagne-temps/health",
            "/gagne-temps/leagues",
            "/gagne-temps/today",
            "/gagne-temps/top",
            "/gagne-temps/predict",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": MODEL_VERSION,
        "predictor_loaded": (
            club_predictor is not None
        ),
    }


@app.get(
    "/gagne-temps/health"
)
async def gagne_temps_health():

    return {
        "status": "ok",
        "service": APP_NAME,
        "application_version": APP_VERSION,
        "version": MODEL_VERSION,
        "predictor_loaded": (
            club_predictor is not None
        ),
        "source": "OpenFootball + TikaML",
        "model": "TikaML MatchPredictor",
        "season": OPENFOOTBALL_SEASON,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# LEAGUES
# ============================================================

@app.get(
    "/gagne-temps/leagues"
)
async def gagne_temps_leagues():

    return {
        "status": "success",
        "season": OPENFOOTBALL_SEASON,
        "leagues": [
            {
                "code": code,
                "name": config["name"],
                "country": config["country"],
            }
            for code, config
            in LEAGUES.items()
        ],
    }


# ============================================================
# TODAY
# ============================================================

@app.get(
    "/gagne-temps/today"
)
async def gagne_temps_today():

    today = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d"
    )

    try:

        return get_predictions_for_date(
            today,
            top_only=False,
        )

    except Exception as exc:

        log.exception(
            "Today endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# TOP
# ============================================================

@app.get(
    "/gagne-temps/top"
)
async def gagne_temps_top():

    today = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d"
    )

    try:

        return get_predictions_for_date(
            today,
            top_only=True,
        )

    except Exception as exc:

        log.exception(
            "Top endpoint failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# PUBLIC GAGNE TEMPS PREDICT
# ============================================================

@app.post(
    "/gagne-temps/predict"
)
async def gagne_temps_predict(
    request: GagneTempsPredictRequest,
):

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "TikaML MatchPredictor "
                "is not loaded"
            ),
        )

    try:

        prediction = (
            predict_gagne_temps_match(
                home_team=request.home_team,
                away_team=request.away_team,
                league=request.league.upper(),
                season=request.season,
                match_date=request.match_date,
                week=request.week,
                kickoff=request.kickoff,
            )
        )

        return clean_json(
            {
                "status": "success",
                "source": "OpenFootball + TikaML",
                "model": "TikaML MatchPredictor",
                "version": MODEL_VERSION,
                "prediction": prediction,
            }
        )

    except Exception as exc:

        log.exception(
            "GAGNE TEMPS prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# PROTECTED DIRECT PREDICT
# ============================================================

@app.post(
    "/predict",
    dependencies=[
        Depends(
            verify_api_key
        )
    ],
)
async def protected_predict(
    request: PredictionRequest,
):

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "TikaML MatchPredictor "
                "is not loaded"
            ),
        )

    try:

        result = safe_tikaml_predict(
            predictor=club_predictor,
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league.upper(),
            season=request.season,
            match_date=request.match_date,
            week=request.week,
            max_goals=MAX_GOALS,
            odds=request.odds,
        )

        return clean_json(
            result
        )

    except Exception as exc:

        log.exception(
            "Protected prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# PROTECTED LIVE
# ============================================================

@app.post(
    "/predict/live",
    dependencies=[
        Depends(
            verify_api_key
        )
    ],
)
async def protected_live_predict(
    payload: dict[str, Any],
):

    if club_predictor is not None and hasattr(
        club_predictor,
        "predict_live",
    ):

        try:

            result = (
                club_predictor.predict_live(
                    **payload
                )
            )

            return clean_json(
                result
            )

        except Exception as exc:

            log.exception(
                "TikaML live prediction failed"
            )

            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    if live_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "LivePredictor is not loaded"
            ),
        )

    try:

        if hasattr(
            live_predictor,
            "predict",
        ):

            result = (
                live_predictor.predict(
                    **payload
                )
            )

        elif hasattr(
            live_predictor,
            "predict_live",
        ):

            result = (
                live_predictor.predict_live(
                    **payload
                )
            )

        else:

            raise RuntimeError(
                "Unsupported LivePredictor interface"
            )

        return clean_json(
            result
        )

    except Exception as exc:

        log.exception(
            "Live prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# BACKFILL
# ============================================================

@app.post(
    "/backfill",
    dependencies=[
        Depends(
            verify_api_key
        )
    ],
)
async def protected_backfill(
    request: PredictionRequest,
):

    if backfill_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Backfill predictor "
                "is not loaded"
            ),
        )

    try:

        result = safe_tikaml_predict(
            predictor=backfill_predictor,
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league.upper(),
            season=request.season,
            match_date=request.match_date,
            week=request.week,
            max_goals=MAX_GOALS,
            odds=request.odds,
        )

        return clean_json(
            result
        )

    except Exception as exc:

        log.exception(
            "Backfill prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# MODEL STATUS
# ============================================================

@app.get(
    "/model-status",
    dependencies=[
        Depends(
            verify_api_key
        )
    ],
)
async def model_status():

    model = None

    if club_predictor is not None:
        model = getattr(
            club_predictor,
            "model",
            None,
        )

    return {
        "service": APP_NAME,
        "application_version": APP_VERSION,
        "model_version": MODEL_VERSION,

        "predictor_loaded": (
            club_predictor is not None
        ),

        "goals_model_loaded": (
            model is not None
        ),

        "predict_lambdas_available": (
            model is not None
            and hasattr(
                model,
                "predict_lambdas",
            )
        ),

        "predict_score_matrix_available": (
            model is not None
            and hasattr(
                model,
                "predict_score_matrix",
            )
        ),

        "generic_model_predict_available": (
            model is not None
            and hasattr(
                model,
                "predict",
            )
        ),

        "corner_model_loaded": (
            club_predictor is not None
            and getattr(
                club_predictor,
                "corner_model",
                None,
            ) is not None
        ),

        "yellow_model_loaded": (
            club_predictor is not None
            and getattr(
                club_predictor,
                "yellow_model",
                None,
            ) is not None
        ),

        "live_predictor_loaded": (
            live_predictor is not None
        ),

        "backfill_predictor_loaded": (
            backfill_predictor is not None
        ),

        "openfootball": True,

        "openfootball_season": (
            OPENFOOTBALL_SEASON
        ),

        "leagues": list(
            LEAGUES.keys()
        ),
    }


# ============================================================
# DEBUG OPENFOOTBALL
# ============================================================

@app.get(
    "/debug/openfootball/{league_code}",
    dependencies=[
        Depends(
            verify_api_key
        )
    ],
)
async def debug_openfootball(
    league_code: str,
):

    league_code = (
        league_code.upper()
    )

    if league_code not in LEAGUES:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown league: "
                f"{league_code}"
            ),
        )

    try:

        matches = (
            fetch_openfootball_league(
                league_code,
                OPENFOOTBALL_SEASON,
            )
        )

        return clean_json(
            {
                "status": "success",
                "league": league_code,
                "season": (
                    OPENFOOTBALL_SEASON
                ),
                "count": len(
                    matches
                ),
                "matches": matches,
            }
        )

    except Exception as exc:

        log.exception(
            "OpenFootball debug failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# LOCAL START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "8001",
            )
        ),
        reload=False,
    )
