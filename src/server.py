"""
GAGNE TEMPS + TikaML Prediction API
------------------------------------

Football prediction API using:
- TikaML MatchPredictor
- OpenFootball current fixtures
- LightGBM + Poisson
- Historical TikaML features
- Current-season OpenFootball fixtures

Public mobile endpoints:
    GET  /health
    GET  /gagne-temps/health
    GET  /gagne-temps/leagues
    GET  /gagne-temps/today
    GET  /gagne-temps/top
    POST /gagne-temps/predict

Protected TikaML endpoints:
    POST /predict
    POST /backfill
    GET  /model-status
    /national/*

Authentication:
    X-API-Key
    TIKA_API_KEY environment variable

Never expose TIKA_API_KEY to the mobile application.
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
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Security,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# TikaML imports
# ---------------------------------------------------------------------

try:
    from src.models.goals import LGBMPoissonModel
except Exception:
    LGBMPoissonModel = None

try:
    from src.models.features import (
        FEATURE_COLS,
        CORNER_FEATURE_COLS,
        YELLOW_FEATURE_COLS,
    )
except Exception:
    FEATURE_COLS = []
    CORNER_FEATURE_COLS = []
    YELLOW_FEATURE_COLS = []

try:
    from src.inference import MatchPredictor
except Exception:
    MatchPredictor = None

try:
    from src.live import LivePredictor
except Exception:
    LivePredictor = None

try:
    from src import national_api
except Exception:
    national_api = None


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("gagne-temps")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_NAME = "GAGNE TEMPS"
APP_VERSION = "2.0.0"

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = MODEL_DIR / "corners"
YELLOW_MODEL_DIR = MODEL_DIR / "yellows"

BACKFILL_DIR = MODEL_DIR / "backfill_20260131"

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "opta" / "processed" / "features.csv"

MAX_GOALS = 7

OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/master"
)

OPENFOOTBALL_SEASON = "2026-27"

OPENFOOTBALL_TIMEOUT = 20

CACHE_TTL_SECONDS = 300


# ---------------------------------------------------------------------
# League configuration
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# API authentication
# ---------------------------------------------------------------------

API_KEY = os.environ.get("TIKA_API_KEY", "").strip()

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(
        "TIKA_API_KEY is not configured. "
        "A temporary key has been generated for this process."
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

    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

class Models:
    goals: Any = None
    corners: Any = None
    yellows: Any = None
    version: str = "unknown"


models = Models()

club_predictor: Any = None
backfill_predictor: Any = None
live_predictor: Any = None


# ---------------------------------------------------------------------
# OpenFootball cache
# ---------------------------------------------------------------------

_openfootball_cache: dict[str, dict[str, Any]] = {}


def _cache_get(key: str) -> Any | None:
    item = _openfootball_cache.get(key)

    if not item:
        return None

    if time.time() - item["timestamp"] > CACHE_TTL_SECONDS:
        _openfootball_cache.pop(key, None)
        return None

    return item["data"]


def _cache_set(key: str, data: Any) -> None:
    _openfootball_cache[key] = {
        "timestamp": time.time(),
        "data": data,
    }


# ---------------------------------------------------------------------
# Team aliases
# ---------------------------------------------------------------------

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
        "Venice",
    ],
    "ACF Fiorentina": [
        "ACF Fiorentina",
        "Fiorentina",
    ],
}


def normalize_team_name(name: str) -> str:
    """
    Normalize team names for robust matching.
    """

    if not name:
        return ""

    value = str(name).strip().lower()

    value = (
        value.replace("’", "'")
        .replace("-", " ")
        .replace("_", " ")
    )

    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def resolve_tika_team(
    openfootball_name: str,
    available_teams: list[str] | None = None,
) -> str:
    """
    Resolve an OpenFootball team name to the name used by TikaML.
    """

    if not openfootball_name:
        raise ValueError("Empty team name")

    candidates = [
        openfootball_name,
    ]

    if openfootball_name in TEAM_ALIASES:
        candidates.extend(
            TEAM_ALIASES[openfootball_name]
        )

    normalized_candidates = {
        normalize_team_name(x)
        for x in candidates
        if x
    }

    if available_teams:
        # Exact normalized match
        for team in available_teams:
            if normalize_team_name(team) in normalized_candidates:
                return team

        # Reverse alias search
        for canonical, aliases in TEAM_ALIASES.items():
            all_names = [canonical] + aliases

            normalized_aliases = {
                normalize_team_name(x)
                for x in all_names
            }

            if normalized_candidates.intersection(
                normalized_aliases
            ):
                for team in available_teams:
                    if normalize_team_name(team) in normalized_aliases:
                        return team

        # Conservative substring match
        for team in available_teams:
            normalized_team = normalize_team_name(team)

            for candidate in normalized_candidates:
                if (
                    candidate in normalized_team
                    or normalized_team in candidate
                ):
                    return team

    # Known direct mappings
    direct = {
        normalize_team_name(
            "1. FC Union Berlin"
        ): "Union Berlin",
        normalize_team_name(
            "FC Schalke 04"
        ): "Schalke 04",
        normalize_team_name(
            "Stade Rennais FC 1901"
        ): "Rennes",
        normalize_team_name(
            "Olympique de Marseille"
        ): "Olympique Marseille",
        normalize_team_name(
            "Sevilla FC"
        ): "Sevilla",
        normalize_team_name(
            "Valencia CF"
        ): "Valencia",
        normalize_team_name(
            "Venezia FC"
        ): "Venezia",
        normalize_team_name(
            "ACF Fiorentina"
        ): "Fiorentina",
    }

    normalized = normalize_team_name(
        openfootball_name
    )

    if normalized in direct:
        return direct[normalized]

    return openfootball_name


# ---------------------------------------------------------------------
# OpenFootball
# ---------------------------------------------------------------------

def openfootball_url(
    league_code: str,
    season: str = OPENFOOTBALL_SEASON,
) -> str:
    config = LEAGUES.get(league_code)

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
    """
    Fetch one OpenFootball league file.
    """

    cache_key = f"{season}:{league_code}"

    cached = _cache_get(cache_key)

    if cached is not None:
        return cached

    url = openfootball_url(
        league_code,
        season,
    )

    try:
        response = requests.get(
            url,
            timeout=OPENFOOTBALL_TIMEOUT,
            headers={
                "User-Agent": (
                    "GAGNE-TEMPS/2.0 "
                    "(football prediction application)"
                )
            },
        )

        response.raise_for_status()

        data = response.json()

        matches: list[dict[str, Any]] = []

        # OpenFootball normally stores games under rounds
        rounds = data.get("rounds", [])

        for round_item in rounds:
            round_name = round_item.get(
                "name",
                round_item.get("round"),
            )

            for match in round_item.get(
                "matches",
                []
            ):
                item = dict(match)

                if round_name:
                    item["_round"] = round_name

                item["_league"] = league_code
                item["_season"] = season

                matches.append(item)

        _cache_set(cache_key, matches)

        return matches

    except Exception as exc:
        log.exception(
            "OpenFootball error for %s",
            league_code,
        )
        raise RuntimeError(
            f"OpenFootball error for {league_code}: {exc}"
        ) from exc


def fetch_all_openfootball(
    season: str = OPENFOOTBALL_SEASON,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    all_matches: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for league_code in LEAGUES:
        try:
            matches = fetch_openfootball_league(
                league_code,
                season,
            )
            all_matches.extend(matches)

        except Exception as exc:
            errors[league_code] = str(exc)

    return all_matches, errors


# ---------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------

def parse_match_date(
    value: Any,
) -> pd.Timestamp | None:
    if value is None:
        return None

    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def normalize_date_string(
    value: Any,
) -> str | None:
    parsed = parse_match_date(value)

    if parsed is None:
        return None

    return parsed.strftime("%Y-%m-%d")


def normalize_season(
    season: str | None,
) -> str:
    """
    Convert:
        2026-27
    into:
        2026-2027
    for TikaML historical data.
    """

    if not season:
        return "2026-2027"

    value = str(season).strip()

    match = re.fullmatch(
        r"(\d{4})-(\d{2}|\d{4})",
        value,
    )

    if not match:
        return value

    first = match.group(1)
    second = match.group(2)

    if len(second) == 2:
        second = first[:2] + second

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

    try:
        return int(match.group(1))
    except Exception:
        return None


def match_has_final_score(
    match: dict[str, Any],
) -> bool:
    score = match.get("score")

    if not score:
        return False

    if isinstance(score, dict):
        ft = score.get("ft")

        if ft is None:
            return False

        if isinstance(ft, (list, tuple)):
            return len(ft) >= 2

        if isinstance(ft, str):
            return bool(ft.strip())

        return True

    return False


# ---------------------------------------------------------------------
# JSON cleaning
# ---------------------------------------------------------------------

def clean_json(value: Any) -> Any:
    """
    Convert NumPy/Pandas objects and NaN/Inf
    into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): clean_json(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            clean_json(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return [
            clean_json(v)
            for v in value.tolist()
        ]

    if isinstance(value, np.generic):
        return clean_json(value.item())

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.DataFrame):
        return clean_json(
            value.to_dict(orient="records")
        )

    if isinstance(value, pd.Series):
        return clean_json(
            value.to_dict()
        )

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, bool):
        return bool(value)

    try:
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)):
            if bool(missing):
                return None
    except Exception:
        pass

    return value


# ---------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------

def calculate_confidence(
    probs: list[float] | np.ndarray,
) -> tuple[float, str]:
    """
    Conservative confidence based on:
    - maximum probability
    - margin over second-best outcome
    """

    values = np.asarray(
        probs,
        dtype=float,
    )

    if values.size != 3:
        return 0.0, "FAIBLE"

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = values.sum()

    if total <= 0:
        return 0.0, "FAIBLE"

    values = values / total

    ordered = np.sort(values)[::-1]

    best = float(ordered[0])
    second = float(ordered[1])

    margin = max(
        0.0,
        best - second,
    )

    # Conservative confidence score
    score = (
        0.55 * best
        + 0.45 * min(
            1.0,
            margin * 2.5,
        )
    )

    confidence = max(
        0.0,
        min(
            100.0,
            score * 100.0,
        ),
    )

    if confidence >= 75:
        label = "FORTE"
    elif confidence >= 60:
        label = "MOYENNE"
    else:
        label = "FAIBLE"

    return round(confidence, 1), label


# ---------------------------------------------------------------------
# TikaML result conversion
# ---------------------------------------------------------------------

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
            [0.0, 0.0, 0.0]
        )

    probs = np.nan_to_num(
        probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = probs.sum()

    if total > 0:
        probs = probs / total

    p_home = float(probs[0])
    p_draw = float(probs[1])
    p_away = float(probs[2])

    sides = [
        "home",
        "draw",
        "away",
    ]

    outcome_idx = int(
        np.argmax(probs)
    )

    predicted_side = sides[outcome_idx]

    confidence, confidence_label = (
        calculate_confidence(probs)
    )

    recommended = result.get(
        "recommended_score",
        {},
    )

    if isinstance(recommended, dict):
        score_label = recommended.get(
            "label"
        )

        if not score_label:
            home_goals = recommended.get(
                "home_goals"
            )
            away_goals = recommended.get(
                "away_goals"
            )

            if (
                home_goals is not None
                and away_goals is not None
            ):
                score_label = (
                    f"{int(home_goals)}"
                    f"-"
                    f"{int(away_goals)}"
                )
    else:
        score_label = str(
            recommended
        )

    if not score_label:
        score_label = "N/A"

    labels = {
        "home": home_team,
        "draw": "Match nul",
        "away": away_team,
    }

    best_pick = labels[
        predicted_side
    ]

    sorted_probs = np.sort(
        probs
    )[::-1]

    margin = float(
        sorted_probs[0]
        - sorted_probs[1]
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "kickoff": kickoff,
        "predicted_side": predicted_side,
        "best_pick": best_pick,
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
            "prob": (
                round(
                    float(
                        recommended.get(
                            "prob",
                            0.0,
                        )
                    ),
                    4,
                )
                if isinstance(
                    recommended,
                    dict,
                )
                else 0.0
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
        "goals_over_under": result.get(
            "goals_over_under",
            {},
        ),
        "corners": result.get(
            "corners",
            {},
        ),
        "yellows": result.get(
            "yellows",
            {},
        ),
    }


# ---------------------------------------------------------------------
# Match prediction
# ---------------------------------------------------------------------

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

    available_teams: list[str] = []

    try:
        if getattr(
            club_predictor,
            "df",
            None,
        ) is not None:

            if (
                "home_team"
                in club_predictor.df.columns
            ):
                available_teams.extend(
                    club_predictor.df[
                        "home_team"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

            if (
                "away_team"
                in club_predictor.df.columns
            ):
                available_teams.extend(
                    club_predictor.df[
                        "away_team"
                    ]
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
    except Exception:
        available_teams = []

    home_tika = resolve_tika_team(
        home_team,
        available_teams,
    )

    away_tika = resolve_tika_team(
        away_team,
        available_teams,
    )

    normalized_season = normalize_season(
        season
    )

    # This is the official TikaML MatchPredictor path.
    result = club_predictor.predict(
        home_team=home_tika,
        away_team=away_tika,
        league=league,
        season=normalized_season,
        match_date=match_date,
        week=week,
        max_goals=MAX_GOALS,
    )

    converted = convert_tikaml_prediction(
        result=result,
        home_team=home_team,
        away_team=away_team,
        league=league,
        kickoff=kickoff,
    )

    converted["tika_home_team"] = home_tika
    converted["tika_away_team"] = away_tika

    return clean_json(
        converted
    )


# ---------------------------------------------------------------------
# Current OpenFootball predictions
# ---------------------------------------------------------------------

def get_predictions_for_date(
    target_date: str,
    top_only: bool = False,
) -> dict[str, Any]:

    matches, errors = (
        fetch_all_openfootball(
            OPENFOOTBALL_SEASON
        )
    )

    predictions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for match in matches:

        match_date = normalize_date_string(
            match.get("date")
        )

        if match_date != target_date:
            continue

        if match_has_final_score(match):
            skipped.append(
                {
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

        home = match.get("team1")
        away = match.get("team2")

        if not home or not away:
            skipped.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "reason": "missing_team",
                }
            )
            continue

        league_code = match.get(
            "_league"
        )

        league_config = LEAGUES.get(
            league_code,
            {},
        )

        league_name = league_config.get(
            "name",
            league_code,
        )

        round_name = match.get(
            "_round"
        )

        week = round_to_week(
            round_name
        )

        try:
            prediction = (
                predict_gagne_temps_match(
                    home_team=home,
                    away_team=away,
                    league=league_code,
                    season=OPENFOOTBALL_SEASON,
                    match_date=match_date,
                    week=week,
                    kickoff=match.get(
                        "time"
                    ),
                )
            )

            prediction["league_code"] = (
                league_code
            )

            prediction["league_name"] = (
                league_name
            )

            prediction["country"] = (
                league_config.get(
                    "country"
                )
            )

            prediction["round"] = (
                round_name
            )

            prediction["date"] = (
                match_date
            )

            prediction["source"] = (
                "OpenFootball + TikaML"
            )

            predictions.append(
                prediction
            )

        except Exception as exc:
            log.exception(
                "Prediction failed: %s vs %s",
                home,
                away,
            )

            skipped.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "league": league_code,
                    "reason": "prediction_error",
                    "error": str(exc),
                }
            )

    # Highest confidence first
    predictions.sort(
        key=lambda item: (
            float(
                item.get(
                    "confidence",
                    0,
                )
            ),
            float(
                item.get(
                    "margin",
                    0,
                )
            ),
        ),
        reverse=True,
    )

    if top_only:
        predictions = predictions[:10]

    return clean_json(
        {
            "date": target_date,
            "season": OPENFOOTBALL_SEASON,
            "source": (
                "OpenFootball + TikaML"
            ),
            "model": (
                "TikaML MatchPredictor"
            ),
            "version": models.version,
            "count": len(predictions),
            "predictions": predictions,
            "skipped": skipped,
            "source_errors": errors,
        }
    )


# ---------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------

class MatchContext(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str = OPENFOOTBALL_SEASON
    match_date: str
    week: int | None = None
    kickoff: str | None = None


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str = OPENFOOTBALL_SEASON
    match_date: str
    week: int | None = None
    odds: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    predicted_outcome: str | None = None
    recommended_score: dict[str, Any] | None = None
    probs_1x2: list[float] | None = None
    lambda_home: float | None = None
    lambda_away: float | None = None
    top_scores: list[Any] | None = None


class GagneTempsPredictRequest(BaseModel):
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
    season: str = OPENFOOTBALL_SEASON
    match_date: str
    week: int | None = None
    kickoff: str | None = None


# ---------------------------------------------------------------------
# Standard TikaML helpers
# ---------------------------------------------------------------------

def _build_feature_df(
    predictor: Any,
    features: dict[str, Any],
) -> pd.DataFrame:

    if predictor is None:
        raise RuntimeError(
            "Predictor unavailable"
        )

    return pd.DataFrame(
        [features]
    )


def _poisson_over_under(
    lambda_home: float,
    lambda_away: float,
) -> dict[str, Any]:

    total_lambda = max(
        0.0,
        float(lambda_home)
        + float(lambda_away),
    )

    result: dict[str, Any] = {}

    try:
        from math import exp, factorial

        probabilities: dict[int, float] = {}

        for goals in range(
            0,
            MAX_GOALS * 2 + 1,
        ):
            p = (
                exp(-total_lambda)
                * total_lambda**goals
                / factorial(goals)
            )

            probabilities[
                goals
            ] = p

        for line in (
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
        ):
            under = sum(
                p
                for goals, p
                in probabilities.items()
                if goals <= line
            )

            over = max(
                0.0,
                1.0 - under,
            )

            result[
                f"{line:.1f}"
            ] = {
                "over": round(
                    over,
                    4,
                ),
                "under": round(
                    under,
                    4,
                ),
            }

    except Exception:
        return {}

    return result


def _goals_over_under(
    result: dict[str, Any],
) -> dict[str, Any]:

    lambda_home = float(
        result.get(
            "lambda_home",
            0.0,
        )
        or 0.0
    )

    lambda_away = float(
        result.get(
            "lambda_away",
            0.0,
        )
        or 0.0
    )

    return _poisson_over_under(
        lambda_home,
        lambda_away,
    )


def _live_over_under(
    lambda_home: float,
    lambda_away: float,
) -> dict[str, Any]:
    return _poisson_over_under(
        lambda_home,
        lambda_away,
    )


# ---------------------------------------------------------------------
# Prematch prediction
# ---------------------------------------------------------------------

def predict_prematch(
    request: PredictionRequest,
) -> dict[str, Any]:

    global club_predictor

    if club_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="TikaML model is not loaded",
        )

    try:
        result = club_predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            season=normalize_season(
                request.season
            ),
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
            "Prematch prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------

def predict_live(
    payload: dict[str, Any],
) -> dict[str, Any]:

    global live_predictor

    if live_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Live predictor is not loaded",
        )

    try:
        if hasattr(
            live_predictor,
            "predict",
        ):
            result = live_predictor.predict(
                **payload
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

    except TypeError:
        # Fallback for different TikaML live signatures.
        try:
            result = live_predictor.predict(
                payload
            )

            return clean_json(
                result
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    except Exception as exc:
        log.exception(
            "Live prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Application startup
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global club_predictor
    global backfill_predictor
    global live_predictor

    log.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    # -------------------------------------------------------------
    # Goals model
    # -------------------------------------------------------------

    if LGBMPoissonModel is not None:
        try:
            if MODEL_DIR.exists():
                models.goals = (
                    LGBMPoissonModel.load(
                        str(MODEL_DIR)
                    )
                )

                log.info(
                    "Main goals model loaded"
                )

        except Exception as exc:
            log.exception(
                "Could not load main goals model: %s",
                exc,
            )

    # -------------------------------------------------------------
    # Corner model
    # -------------------------------------------------------------

    if (
        LGBMPoissonModel is not None
        and CORNER_MODEL_DIR.exists()
    ):
        try:
            models.corners = (
                LGBMPoissonModel.load(
                    str(CORNER_MODEL_DIR)
                )
            )

            log.info(
                "Corner model loaded"
            )

        except Exception as exc:
            log.warning(
                "Corner model unavailable: %s",
                exc,
            )

    # -------------------------------------------------------------
    # Yellow-card model
    # -------------------------------------------------------------

    if (
        LGBMPoissonModel is not None
        and YELLOW_MODEL_DIR.exists()
    ):
        try:
            models.yellows = (
                LGBMPoissonModel.load(
                    str(YELLOW_MODEL_DIR)
                )
            )

            log.info(
                "Yellow-card model loaded"
            )

        except Exception as exc:
            log.warning(
                "Yellow model unavailable: %s",
                exc,
            )

    # -------------------------------------------------------------
    # Model metadata
    # -------------------------------------------------------------

    meta_file = MODEL_DIR / "meta.json"

    if meta_file.exists():
        try:
            with meta_file.open(
                "r",
                encoding="utf-8",
            ) as f:
                meta = json.load(f)

            feature_count = len(
                meta.get(
                    "feature_cols",
                    [],
                )
            )

            if feature_count:
                models.version = (
                    f"lgbm-poisson-"
                    f"{feature_count}f"
                )
            else:
                models.version = (
                    meta.get(
                        "version",
                        "unknown",
                    )
                )

        except Exception as exc:
            log.warning(
                "Could not read model metadata: %s",
                exc,
            )

    # -------------------------------------------------------------
    # TikaML MatchPredictor
    # -------------------------------------------------------------

    if MatchPredictor is not None:
        try:
            club_predictor = (
                MatchPredictor()
            )

            club_predictor.load_model()

            log.info(
                "TikaML MatchPredictor loaded"
            )

        except Exception as exc:
            club_predictor = None

            log.exception(
                "Could not load MatchPredictor: %s",
                exc,
            )

    # -------------------------------------------------------------
    # Live predictor
    # -------------------------------------------------------------

    if LivePredictor is not None:
        try:
            live_predictor = (
                LivePredictor()
            )

            # If LivePredictor has a load method
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

    # -------------------------------------------------------------
    # Backfill predictor
    # -------------------------------------------------------------

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

            if backfill_goals.exists():
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

    # -------------------------------------------------------------
    # National API
    # -------------------------------------------------------------

    if national_api is not None:
        log.info(
            "National API module detected"
        )

    log.info(
        "%s startup completed",
        APP_NAME,
    )

    yield

    log.info(
        "%s shutting down",
        APP_NAME,
    )


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title=APP_NAME,
    description=(
        "Football prediction API powered by "
        "TikaML and OpenFootball."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# National protected router
# ---------------------------------------------------------------------

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
            "Could not register national router: %s",
            exc,
        )


# ---------------------------------------------------------------------
# Public health
# ---------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "source": (
            "OpenFootball + TikaML"
        ),
        "season": OPENFOOTBALL_SEASON,
        "model": (
            "TikaML MatchPredictor"
        ),
        "model_version": models.version,
        "public_endpoints": [
            "/health",
            "/gagne-temps/health",
            "/gagne-temps/leagues",
            "/gagne-temps/today",
            "/gagne-temps/top",
            "/gagne-temps/predict",
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": (
            club_predictor is not None
        ),
        "model_version": models.version,
        "season": OPENFOOTBALL_SEASON,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


@app.get("/gagne-temps/health")
async def gagne_temps_health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "model": (
            "TikaML MatchPredictor"
        ),
        "model_loaded": (
            club_predictor is not None
        ),
        "model_version": models.version,
        "openfootball": True,
        "season": OPENFOOTBALL_SEASON,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ---------------------------------------------------------------------
# Public leagues
# ---------------------------------------------------------------------

@app.get("/gagne-temps/leagues")
async def gagne_temps_leagues():
    return {
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


# ---------------------------------------------------------------------
# Public today's matches
# ---------------------------------------------------------------------

@app.get("/gagne-temps/today")
async def gagne_temps_today():
    """
    Return all upcoming matches for today's UTC date.
    """

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    try:
        result = get_predictions_for_date(
            today,
            top_only=False,
        )

        return result

    except Exception as exc:
        log.exception(
            "Today endpoint error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Public top predictions
# ---------------------------------------------------------------------

@app.get("/gagne-temps/top")
async def gagne_temps_top():
    """
    Return highest-confidence upcoming
    predictions for today.
    """

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    try:
        return get_predictions_for_date(
            today,
            top_only=True,
        )

    except Exception as exc:
        log.exception(
            "Top endpoint error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Public mobile prediction
# ---------------------------------------------------------------------

@app.post("/gagne-temps/predict")
async def gagne_temps_predict(
    request: GagneTempsPredictRequest,
):
    """
    Public prediction endpoint for the
    GAGNE TEMPS mobile application.

    No API key required.
    """

    if club_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="TikaML model is not loaded",
        )

    try:
        result = predict_gagne_temps_match(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            season=request.season,
            match_date=request.match_date,
            week=request.week,
            kickoff=request.kickoff,
        )

        return clean_json(
            {
                "success": True,
                "source": (
                    "OpenFootball + TikaML"
                ),
                "model": (
                    "TikaML MatchPredictor"
                ),
                "version": models.version,
                "prediction": result,
            }
        )

    except Exception as exc:
        log.exception(
            "GAGNE TEMPS prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Protected standard TikaML prediction
# ---------------------------------------------------------------------

@app.post(
    "/predict",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def protected_predict(
    request: PredictionRequest,
):
    """
    Protected direct TikaML endpoint.
    """

    return predict_prematch(
        request
    )


# ---------------------------------------------------------------------
# Protected live prediction
# ---------------------------------------------------------------------

@app.post(
    "/predict/live",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def protected_live_predict(
    payload: dict[str, Any],
):
    return predict_live(
        payload
    )


# ---------------------------------------------------------------------
# Protected backfill
# ---------------------------------------------------------------------

@app.post(
    "/backfill",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def backfill_predict(
    request: PredictionRequest,
):

    global backfill_predictor

    if backfill_predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Backfill predictor is not loaded"
            ),
        )

    try:
        result = backfill_predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            league=request.league,
            season=normalize_season(
                request.season
            ),
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
            "Backfill prediction error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Protected model status
# ---------------------------------------------------------------------

@app.get(
    "/model-status",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def model_status():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "model_version": models.version,
        "goals_model": (
            models.goals is not None
        ),
        "corners_model": (
            models.corners is not None
        ),
        "yellows_model": (
            models.yellows is not None
        ),
        "match_predictor": (
            club_predictor is not None
        ),
        "live_predictor": (
            live_predictor is not None
        ),
        "backfill_predictor": (
            backfill_predictor is not None
        ),
        "openfootball_season": (
            OPENFOOTBALL_SEASON
        ),
        "leagues": list(
            LEAGUES.keys()
        ),
    }


# ---------------------------------------------------------------------
# Debug endpoint - protected
# ---------------------------------------------------------------------

@app.get(
    "/debug/openfootball/{league_code}",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def debug_openfootball(
    league_code: str,
):

    league_code = league_code.upper()

    if league_code not in LEAGUES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown league: "
                f"{league_code}"
            ),
        )

    try:
        matches = fetch_openfootball_league(
            league_code,
            OPENFOOTBALL_SEASON,
        )

        return {
            "league": league_code,
            "season": OPENFOOTBALL_SEASON,
            "count": len(matches),
            "matches": clean_json(
                matches
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Error handler for unexpected exceptions
# ---------------------------------------------------------------------

@app.exception_handler(
    ValueError
)
async def value_error_handler(
    request,
    exc: ValueError,
):
    return HTTPException(
        status_code=400,
        detail=str(exc),
    )


# ---------------------------------------------------------------------
# Local execution
# ---------------------------------------------------------------------

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
