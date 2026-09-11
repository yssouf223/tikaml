"""
GAGNE TEMPS + TikaML Prediction API
===================================

Football prediction API based on:

    OpenFootball
    +
    TikaML MatchPredictor

IMPORTANT:
Do NOT call:

    LGBMPoissonModel.predict()

The TikaML MatchPredictor is used instead.

Public endpoints:
    GET  /
    GET  /health
    GET  /gagne-temps/health
    GET  /gagne-temps/leagues
    GET  /gagne-temps/today
    GET  /gagne-temps/top
    POST /gagne-temps/predict

Protected endpoints:
    POST /predict
    GET  /model-status
    GET  /debug/openfootball/{league_code}

Environment:
    TIKA_API_KEY

Server:
    Uvicorn / Render
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

from src.inference import MatchPredictor


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

log = logging.getLogger(
    "gagne-temps"
)


# ============================================================
# APPLICATION CONFIG
# ============================================================

APP_NAME = "GAGNE TEMPS"

APP_VERSION = "3.1.0"

MAX_GOALS = 7

MODEL_DIR = Path(
    "models"
)

FEATURES_FILE = Path(
    "data/opta/processed/features.csv"
)

OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)

OPENFOOTBALL_SEASON = "2026-27"

OPENFOOTBALL_TIMEOUT = 30

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

TIKA_API_KEY = os.environ.get(
    "TIKA_API_KEY",
    "",
).strip()


if not TIKA_API_KEY:

    TIKA_API_KEY = secrets.token_urlsafe(
        32
    )

    log.warning(
        "TIKA_API_KEY absent. "
        "Temporary API key generated."
    )


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str | None = Security(
        api_key_header
    ),
):
    """
    Protect private endpoints.
    """

    if not key:

        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    if not secrets.compare_digest(
        key,
        TIKA_API_KEY,
    ):

        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ============================================================
# GLOBAL TIKAML PREDICTOR
# ============================================================

club_predictor: MatchPredictor | None = None

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

    item = _openfootball_cache.get(
        key
    )

    if item is None:
        return None

    age = (
        time.time()
        - item["timestamp"]
    )

    if age > CACHE_TTL_SECONDS:

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


def cache_clear() -> None:

    _openfootball_cache.clear()


# ============================================================
# OPENFOOTBALL URL
# ============================================================

def openfootball_url(
    league_code: str,
    season: str = OPENFOOTBALL_SEASON,
) -> str:

    config = LEAGUES.get(
        league_code
    )

    if config is None:

        raise ValueError(
            f"Unknown league: "
            f"{league_code}"
        )

    return (
        f"{OPENFOOTBALL_BASE}/"
        f"{season}/"
        f"{config['file']}"
    )


# ============================================================
# TEAM NORMALIZATION
# ============================================================

def normalize_team_name(
    name: str,
) -> str:

    if not name:

        return ""

    value = str(
        name
    ).strip().lower()

    value = (
        value
        .replace(
            "’",
            "'",
        )
        .replace(
            "-",
            " ",
        )
        .replace(
            "_",
            " ",
        )
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


# ============================================================
# TEAM ALIASES
# ============================================================

TEAM_ALIASES: dict[
    str,
    list[str],
] = {

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

    "1 fc union berlin":
        "Union Berlin",

    "fc schalke 04":
        "Schalke 04",

    "stade rennais fc 1901":
        "Rennes",

    "olympique de marseille":
        "Olympique Marseille",

    "sevilla fc":
        "Sevilla",

    "valencia cf":
        "Valencia",

    "venezia fc":
        "Venezia",

    "acf fiorentina":
        "Fiorentina",
}


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

    # --------------------------------------------------------
    # DIRECT ALIAS
    # --------------------------------------------------------

    direct = (
        DIRECT_TIKA_ALIASES.get(
            normalized
        )
    )

    if direct:

        if available_teams:

            normalized_available = {
                normalize_team_name(team):
                team
                for team in available_teams
            }

            if (
                normalize_team_name(
                    direct
                )
                in normalized_available
            ):

                return normalized_available[
                    normalize_team_name(
                        direct
                    )
                ]

        return direct

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    candidates = [
        openfootball_name
    ]

    candidates.extend(
        TEAM_ALIASES.get(
            openfootball_name,
            [],
        )
    )

    normalized_candidates = [
        normalize_team_name(
            item
        )
        for item in candidates
        if item
    ]

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    if available_teams:

        normalized_available = {
            normalize_team_name(team):
            team
            for team in available_teams
        }

        for candidate in normalized_candidates:

            if (
                candidate
                in normalized_available
            ):

                return normalized_available[
                    candidate
                ]

    # --------------------------------------------------------
    # PARTIAL MATCH
    # --------------------------------------------------------

    if available_teams:

        for team in available_teams:

            normalized_team = (
                normalize_team_name(
                    team
                )
            )

            for candidate in normalized_candidates:

                if not candidate:
                    continue

                if (
                    candidate
                    in normalized_team
                    or normalized_team
                    in candidate
                ):

                    return team

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return openfootball_name


# ============================================================
# OPENFOOTBALL FETCH
# ============================================================

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

        log.info(
            "OpenFootball cache HIT: %s",
            league_code,
        )

        return cached

    url = openfootball_url(
        league_code,
        season,
    )

    log.info(
        "OpenFootball GET: %s",
        url,
    )

    response = requests.get(
        url,
        timeout=OPENFOOTBALL_TIMEOUT,
        headers={
            "User-Agent":
                "GAGNE-TEMPS/3.1",
            "Accept":
                "application/json",
        },
    )

    response.raise_for_status()

    data = response.json()

    matches: list[
        dict[str, Any]
    ] = []

    # ========================================================
    # FORMAT A
    # matches directly at root
    # ========================================================

    root_matches = data.get(
        "matches"
    ) if isinstance(
        data,
        dict,
    ) else None

    if isinstance(
        root_matches,
        list,
    ):

        for match in root_matches:

            if not isinstance(
                match,
                dict,
            ):
                continue

            item = dict(
                match
            )

            item["_league"] = (
                league_code
            )

            item["_season"] = (
                season
            )

            item["_round"] = (
                match.get(
                    "round"
                )
                or match.get(
                    "matchday"
                )
            )

            matches.append(
                item
            )

    # ========================================================
    # FORMAT B
    # rounds -> matches
    # ========================================================

    rounds = (
        data.get(
            "rounds"
        )
        if isinstance(
            data,
            dict,
        )
        else None
    )

    if isinstance(
        rounds,
        list,
    ):

        for round_item in rounds:

            if not isinstance(
                round_item,
                dict,
            ):
                continue

            round_name = (
                round_item.get(
                    "name"
                )
                or round_item.get(
                    "round"
                )
                or round_item.get(
                    "matchday"
                )
            )

            round_matches = (
                round_item.get(
                    "matches",
                    [],
                )
            )

            if not isinstance(
                round_matches,
                list,
            ):
                continue

            for match in round_matches:

                if not isinstance(
                    match,
                    dict,
                ):
                    continue

                item = dict(
                    match
                )

                item["_league"] = (
                    league_code
                )

                item["_season"] = (
                    season
                )

                item["_round"] = (
                    round_name
                )

                matches.append(
                    item
                )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    unique_matches = []

    seen = set()

    for match in matches:

        key = (
            match.get("date"),
            match.get("time"),
            match.get("team1"),
            match.get("team2"),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_matches.append(
            match
        )

    matches = unique_matches

    # ========================================================
    # LOG
    # ========================================================

    if not matches:

        log.warning(
            "OpenFootball returned 0 matches "
            "for %s",
            league_code,
        )

        if isinstance(
            data,
            dict,
        ):

            log.warning(
                "OpenFootball root keys: %s",
                list(
                    data.keys()
                ),
            )

    else:

        log.info(
            "OpenFootball %s: %d matches",
            league_code,
            len(matches),
        )

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

    errors: dict[
        str,
        str,
    ] = {}

    for league_code in LEAGUES:

        try:

            league_matches = (
                fetch_openfootball_league(
                    league_code,
                    season,
                )
            )

            all_matches.extend(
                league_matches
            )

        except Exception as exc:

            log.exception(
                "OpenFootball error "
                "%s",
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
# DATE HELPERS
# ============================================================

def normalize_date(
    value: Any,
) -> str | None:

    if value is None:
        return None

    try:

        return pd.Timestamp(
            value
        ).strftime(
            "%Y-%m-%d"
        )

    except Exception:

        return None


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

    first = match.group(
        1
    )

    second = match.group(
        2
    )

    if len(second) == 2:

        second = (
            first[:2]
            + second
        )

    return (
        f"{first}-{second}"
    )


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


# ============================================================
# SCORE HELPERS
# ============================================================

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

    if isinstance(
        ft,
        dict,
    ):

        return (
            ft.get("home") is not None
            and ft.get("away") is not None
        )

    return False


# ============================================================
# JSON CLEANER
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
            str(k):
                clean_json(v)
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
        float,
    ):

        if not np.isfinite(
            value
        ):

            return None

        return float(
            value
        )

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):

        return int(
            value
        )

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):

        return bool(
            value
        )

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

            if bool(
                missing
            ):

                return None

    except Exception:
        pass

    return value


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    probs: list[float]
    | np.ndarray,
) -> tuple[
    float,
    str,
]:

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
        +
        0.45
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

    percentage = round(
        confidence * 100,
        1,
    )

    if percentage >= 75:

        label = "FORTE"

    elif percentage >= 60:

        label = "MOYENNE"

    else:

        label = "FAIBLE"

    return (
        percentage,
        label,
    )


# ============================================================
# TIKAML RESULT CONVERTER
# ============================================================

def convert_tikaml_prediction(
    result: dict[str, Any],
    home_team: str,
    away_team: str,
    league: str,
    kickoff: str | None = None,
) -> dict[str, Any]:

    probs = np.asarray(
        result.get(
            "probs_1x2",
            [
                0.0,
                0.0,
                0.0,
            ],
        ),
        dtype=float,
    )

    if probs.size != 3:

        probs = np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
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

    if predicted_side == "home":

        best_pick = home_team

    elif predicted_side == "away":

        best_pick = away_team

    else:

        best_pick = "Match nul"

    confidence, confidence_label = (
        calculate_confidence(
            probs
        )
    )

    recommended = result.get(
        "recommended_score",
        {},
    )

    if isinstance(
        recommended,
        dict,
    ):

        score_label = (
            recommended.get(
                "label"
            )
        )

        if not score_label:

            hg = (
                recommended.get(
                    "home_goals"
                )
            )

            ag = (
                recommended.get(
                    "away_goals"
                )
            )

            if (
                hg is not None
                and ag is not None
            ):

                score_label = (
                    f"{int(hg)}-"
                    f"{int(ag)}"
                )

        score_probability = (
            recommended.get(
                "prob",
                0.0,
            )
        )

    else:

        score_label = str(
            recommended
        )

        score_probability = 0.0

    if not score_label:

        score_label = "N/A"

    ordered = np.sort(
        probs
    )[::-1]

    margin = float(
        ordered[0]
        - ordered[1]
    )

    return clean_json(
        {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "kickoff": kickoff,

            "best_pick": best_pick,

            "predicted_side":
                predicted_side,

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

                "label":
                    score_label,

                "prob": round(
                    float(
                        score_probability
                    ),
                    4,
                ),
            },

            "confidence":
                confidence,

            "confidence_label":
                confidence_label,

            "margin": round(
                margin,
                4,
            ),

            "lambda_home":
                result.get(
                    "lambda_home"
                ),

            "lambda_away":
                result.get(
                    "lambda_away"
                ),

            "top_scores":
                result.get(
                    "top_scores",
                    [],
                ),

            "score_groups":
                result.get(
                    "score_groups",
                    [],
                ),

            "goals_over_under":
                result.get(
                    "goals_over_under",
                    {},
                ),

            "corners":
                result.get(
                    "corners"
                ),

            "yellows":
                result.get(
                    "yellows"
                ),
        }
    )


# ============================================================
# TIKAML PREDICTION
# ============================================================

def predict_gagne_temps_match(
    home_team: str,
    away_team: str,
    league: str,
    season: str,
    match_date: str,
    week: int | None = None,
    kickoff: str | None = None,
    odds: dict[
        str,
        float,
    ] | None = None,
) -> dict[str, Any]:

    global club_predictor

    if club_predictor is None:

        raise RuntimeError(
            "TikaML MatchPredictor "
            "is not loaded"
        )

    # --------------------------------------------------------
    # Get TikaML teams
    # --------------------------------------------------------

    available_teams: list[
        str
    ] = []

    try:

        df = getattr(
            club_predictor,
            "df",
            None,
        )

        if df is not None:

            for column in (
                "home_team",
                "away_team",
            ):

                if (
                    column
                    in df.columns
                ):

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
            "Unable to read "
            "TikaML teams: %s",
            exc,
        )

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
        "Prediction: "
        "%s [%s] vs %s [%s] | "
        "league=%s | date=%s",
        home_team,
        home_tika,
        away_team,
        away_tika,
        league,
        match_date,
    )

    # ========================================================
    # CRITICAL TIKAML CALL
    # ========================================================
    #
    # This is intentional:
    #
    #     club_predictor.predict(...)
    #
    # NOT:
    #
    #     club_predictor.model.predict(...)
    #
    # ========================================================

    result = club_predictor.predict(
        home_team=home_tika,
        away_team=away_tika,
        league=league,
        season=normalized_season,
        match_date=match_date,
        week=week,
        max_goals=MAX_GOALS,
        odds=odds,
    )

    prediction = (
        convert_tikaml_prediction(
            result=result,
            home_team=home_team,
            away_team=away_team,
            league=league,
            kickoff=kickoff,
        )
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
# PREDICTIONS FOR DATE
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

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    log.info(
        "Searching matches for date: %s",
        target_date,
    )

    log.info(
        "Total OpenFootball matches: %d",
        len(matches),
    )

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    for match in matches:

        match_date = (
            normalize_date(
                match.get(
                    "date"
                )
            )
        )

        if match_date != target_date:

            continue

        # ----------------------------------------------------
        # Ignore completed games
        # ----------------------------------------------------

        if match_has_final_score(
            match
        ):

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

        kickoff = match.get(
            "time"
        )

        if (
            not home_team
            or not away_team
        ):

            skipped.append(
                {
                    "league":
                        league_code,

                    "home_team":
                        home_team,

                    "away_team":
                        away_team,

                    "reason":
                        "missing_team",
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
                    season=(
                        OPENFOOTBALL_SEASON
                    ),
                    match_date=match_date,
                    week=week,
                    kickoff=kickoff,
                )
            )

            config = LEAGUES[
                league_code
            ]

            prediction.update(
                {

                    "league_code":
                        league_code,

                    "league_name":
                        config["name"],

                    "country":
                        config["country"],

                    "round":
                        round_name,

                    "date":
                        match_date,

                    "source":
                        "OpenFootball + TikaML",

                    "model":
                        "TikaML MatchPredictor",

                    "version":
                        MODEL_VERSION,
                }
            )

            predictions.append(
                prediction
            )

            log.info(
                "Prediction OK: "
                "%s vs %s",
                home_team,
                away_team,
            )

        except Exception as exc:

            log.exception(
                "Prediction failed: "
                "%s vs %s",
                home_team,
                away_team,
            )

            skipped.append(
                {

                    "league":
                        league_code,

                    "home_team":
                        home_team,

                    "away_team":
                        away_team,

                    "reason":
                        str(exc),
                }
            )

    # --------------------------------------------------------
    # SORT
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

    result = {

        "status":
            "success",

        "source":
            "OpenFootball + TikaML",

        "model":
            "TikaML MatchPredictor",

        "version":
            MODEL_VERSION,

        "date":
            target_date,

        "count":
            (
                len(top_5)
                if top_only
                else len(predictions)
            ),

        "top_5":
            top_5,

        "skipped":
            skipped,

        "source_errors":
            source_errors,
    }

    if not top_only:

        result[
            "predictions"
        ] = predictions

    return clean_json(
        result
    )


# ============================================================
# REQUEST MODELS
# ============================================================

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

    odds: dict[
        str,
        float,
    ] | None = None


class ProtectedPredictionRequest(
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


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global club_predictor
    global MODEL_VERSION

    log.info(
        "========================================"
    )

    log.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    # ========================================================
    # LOAD TIKAML
    # ========================================================

    try:

        log.info(
            "Loading TikaML MatchPredictor..."
        )

        club_predictor = (
            MatchPredictor()
        )

        club_predictor.load_model()

        if (
            club_predictor.model
            is None
        ):

            raise RuntimeError(
                "TikaML goals model "
                "is None"
            )

        log.info(
            "TikaML MatchPredictor loaded."
        )

        log.info(
            "Goals model loaded."
        )

        if getattr(
            club_predictor,
            "corner_model",
            None,
        ) is not None:

            log.info(
                "Corners model loaded."
            )

        if getattr(
            club_predictor,
            "yellow_model",
            None,
        ) is not None:

            log.info(
                "Yellow cards model loaded."
            )

    except Exception as exc:

        club_predictor = None

        log.exception(
            "TikaML loading failed: %s",
            exc,
        )

    # ========================================================
    # MODEL VERSION
    # ========================================================

    meta_file = (
        MODEL_DIR
        / "meta.json"
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

            feature_cols = meta.get(
                "feature_cols",
                [],
            )

            if feature_cols:

                MODEL_VERSION = (
                    "lgbm-poisson-"
                    f"{len(feature_cols)}f"
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
                "Unable to read "
                "models/meta.json: %s",
                exc,
            )

    if (
        MODEL_VERSION
        == "unknown"
    ):

        MODEL_VERSION = (
            "lgbm-poisson"
        )

    log.info(
        "Model version: %s",
        MODEL_VERSION,
    )

    log.info(
        "Predictor loaded: %s",
        club_predictor
        is not None,
    )

    log.info(
        "========================================"
    )

    yield

    log.info(
        "GAGNE TEMPS shutdown."
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "Football prediction API "
        "powered by TikaML + OpenFootball"
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
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {

        "service":
            APP_NAME,

        "application_version":
            APP_VERSION,

        "status":
            "online",

        "source":
            "OpenFootball + TikaML",

        "model":
            "TikaML MatchPredictor",

        "model_version":
            MODEL_VERSION,

        "predictor_loaded":
            (
                club_predictor
                is not None
            ),

        "season":
            OPENFOOTBALL_SEASON,

        "leagues":
            list(
                LEAGUES.keys()
            ),

        "endpoints": {

            "health":
                "/health",

            "top":
                "/gagne-temps/top",

            "today":
                "/gagne-temps/today",

            "predict":
                "/gagne-temps/predict",

            "leagues":
                "/gagne-temps/leagues",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {

        "status":
            "ok",

        "service":
            APP_NAME,

        "version":
            MODEL_VERSION,

        "predictor_loaded":
            (
                club_predictor
                is not None
            ),
    }


@app.get(
    "/gagne-temps/health"
)
async def gagne_temps_health():

    return {

        "status":
            "ok",

        "service":
            APP_NAME,

        "application_version":
            APP_VERSION,

        "model_version":
            MODEL_VERSION,

        "predictor_loaded":
            (
                club_predictor
                is not None
            ),

        "source":
            "OpenFootball + TikaML",

        "model":
            "TikaML MatchPredictor",

        "season":
            OPENFOOTBALL_SEASON,

        "timestamp":
            datetime.now(
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

        "status":
            "success",

        "season":
            OPENFOOTBALL_SEASON,

        "leagues": [

            {
                "code":
                    code,

                "name":
                    config["name"],

                "country":
                    config["country"],
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

        return (
            get_predictions_for_date(
                today,
                top_only=False,
            )
        )

    except Exception as exc:

        log.exception(
            "Today endpoint failed."
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

        return (
            get_predictions_for_date(
                today,
                top_only=True,
            )
        )

    except Exception as exc:

        log.exception(
            "Top endpoint failed."
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# PUBLIC PREDICTION
# ============================================================

@app.post(
    "/gagne-temps/predict"
)
async def gagne_temps_predict(
    request:
        GagneTempsPredictRequest,
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
                home_team=
                    request.home_team,

                away_team=
                    request.away_team,

                league=
                    request.league,

                season=
                    request.season,

                match_date=
                    request.match_date,

                week=
                    request.week,

                kickoff=
                    request.kickoff,

                odds=
                    request.odds,
            )
        )

        return clean_json(
            {

                "status":
                    "success",

                "source":
                    "OpenFootball + TikaML",

                "model":
                    "TikaML MatchPredictor",

                "version":
                    MODEL_VERSION,

                "prediction":
                    prediction,
            }
        )

    except Exception as exc:

        log.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# PROTECTED PREDICTION
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
    request:
        ProtectedPredictionRequest,
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

        result = (
            club_predictor.predict(
                home_team=
                    request.home_team,

                away_team=
                    request.away_team,

                league=
                    request.league,

                season=
                    normalize_season(
                        request.season
                    ),

                match_date=
                    request.match_date,

                week=
                    request.week,

                max_goals=
                    MAX_GOALS,

                odds=
                    request.odds,
            )
        )

        return clean_json(
            {

                "status":
                    "success",

                "source":
                    "TikaML",

                "model":
                    "TikaML MatchPredictor",

                "version":
                    MODEL_VERSION,

                "prediction":
                    result,
            }
        )

    except Exception as exc:

        log.exception(
            "Protected prediction failed."
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

    return {

        "service":
            APP_NAME,

        "application_version":
            APP_VERSION,

        "model_version":
            MODEL_VERSION,

        "predictor_loaded":
            (
                club_predictor
                is not None
            ),

        "goals_model_loaded":
            (
                club_predictor
                is not None
                and getattr(
                    club_predictor,
                    "model",
                    None,
                )
                is not None
            ),

        "corners_model_loaded":
            (
                club_predictor
                is not None
                and getattr(
                    club_predictor,
                    "corner_model",
                    None,
                )
                is not None
            ),

        "yellows_model_loaded":
            (
                club_predictor
                is not None
                and getattr(
                    club_predictor,
                    "yellow_model",
                    None,
                )
                is not None
            ),

        "openfootball":
            True,

        "openfootball_season":
            OPENFOOTBALL_SEASON,

        "leagues":
            list(
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

    if (
        league_code
        not in LEAGUES
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown league: "
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

                "status":
                    "success",

                "league":
                    league_code,

                "season":
                    OPENFOOTBALL_SEASON,

                "count":
                    len(matches),

                "matches":
                    matches,
            }
        )

    except Exception as exc:

        log.exception(
            "OpenFootball debug failed."
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8001",
        )
    )

    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
