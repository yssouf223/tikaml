"""
GAGNE TEMPS - Prediction API

Architecture:
    OpenFootball
        ↓
    TikaML MatchPredictor
        ↓
    Probabilités / Poisson
        ↓
    GAGNE TEMPS Selection Engine
        ↓
    PREMIUM / BON PRONOSTIC / RISQUÉ / À ÉVITER

Routes publiques:
    GET / 
    GET /health
    GET /gagne-temps/health
    GET /gagne-temps/leagues
    GET /gagne-temps/today
    GET /gagne-temps/top
    GET /gagne-temps/predict

Routes protégées:
    POST /predict
    GET /model-status
    GET /debug/openfootball/{league_code}
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.inference import MatchPredictor
from src.selection_engine import select_prediction


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "GAGNE TEMPS"

OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/"
    "master/2026-27"
)

MAX_GOALS = 7

REQUEST_TIMEOUT = 20

CACHE_TTL = 300


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("gagne-temps")


# ============================================================
# API KEY
# ============================================================

API_KEY = os.environ.get("TIKA_API_KEY", "")

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(
        "TIKA_API_KEY non configurée. "
        "Une clé temporaire a été générée."
    )

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str = Security(api_key_header),
):
    if not key or not secrets.compare_digest(
        key,
        API_KEY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="GAGNE TEMPS",
    description=(
        "Football prediction API powered by "
        "TikaML + OpenFootball + GAGNE TEMPS Selection Engine"
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# TIKAML
# ============================================================

club_predictor: Optional[MatchPredictor] = None

MODEL_VERSION = "unknown"


# ============================================================
# LEAGUES
# ============================================================

LEAGUES = {
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
# ALIAS EQUIPES
# ============================================================

TEAM_ALIASES = {

    "1. FC Union Berlin": [
        "Union Berlin",
        "1. FC Union Berlin",
    ],

    "FC Schalke 04": [
        "Schalke 04",
        "FC Schalke 04",
    ],

    "Stade Rennais FC 1901": [
        "Rennes",
        "Stade Rennais",
        "Stade Rennais FC 1901",
    ],

    "Olympique de Marseille": [
        "Olympique Marseille",
        "Marseille",
        "Olympique de Marseille",
    ],

    "Sevilla FC": [
        "Sevilla",
        "Sevilla FC",
    ],

    "Valencia CF": [
        "Valencia",
        "Valencia CF",
    ],

    "Venezia FC": [
        "Venezia",
        "Venezia FC",
    ],

    "ACF Fiorentina": [
        "Fiorentina",
        "ACF Fiorentina",
    ],

    "FC Internazionale Milano": [
        "Inter",
        "Internazionale",
        "Inter Milan",
    ],

    "AS Roma": [
        "Roma",
        "AS Roma",
    ],

    "AC Milan": [
        "Milan",
        "AC Milan",
    ],

    "SSC Napoli": [
        "Napoli",
        "SSC Napoli",
    ],

    "Juventus FC": [
        "Juventus",
        "Juventus FC",
    ],

    "Lazio": [
        "Lazio",
        "SS Lazio",
    ],

    "Atalanta BC": [
        "Atalanta",
        "Atalanta BC",
    ],

    "Bologna FC 1909": [
        "Bologna",
        "Bologna FC 1909",
    ],

    "Torino FC": [
        "Torino",
        "Torino FC",
    ],

    "Genoa CFC": [
        "Genoa",
        "Genoa CFC",
    ],

    "US Lecce": [
        "Lecce",
        "US Lecce",
    ],

    "Cagliari Calcio": [
        "Cagliari",
        "Cagliari Calcio",
    ],

    "Como 1907": [
        "Como",
        "Como 1907",
    ],
}


# ============================================================
# CACHE OPENFOOTBALL
# ============================================================

_openfootball_cache: dict[str, dict[str, Any]] = {}


# ============================================================
# OUTILS GENERAUX
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


def normalize_probability(
    value: Any,
) -> float:

    if isinstance(value, str):
        value = value.replace("%", "").strip()

    value = safe_float(value)

    if value > 1:
        value /= 100

    return max(
        0.0,
        min(1.0, value),
    )


def percentage(
    value: Any,
) -> float:

    return round(
        normalize_probability(value) * 100,
        1,
    )


def normalize_team_name(
    name: str,
) -> str:

    if not name:
        return ""

    value = name.lower().strip()

    value = re.sub(
        r"[^a-z0-9à-ÿ ]",
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
# SAISON
# ============================================================

def current_season() -> str:
    """
    Saison actuelle utilisée par OpenFootball/TikaML.
    """

    return "2026-2027"


# ============================================================
# MODELE
# ============================================================

def load_predictor() -> None:

    global club_predictor
    global MODEL_VERSION

    log.info(
        "Chargement du modèle TikaML..."
    )

    predictor = MatchPredictor()

    predictor.load_model()

    club_predictor = predictor

    try:

        meta_path = Path(
            "models/meta.json"
        )

        if meta_path.exists():

            with open(
                meta_path,
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

            MODEL_VERSION = (
                f"lgbm-poisson-{feature_count}f"
            )

        else:

            MODEL_VERSION = (
                "lgbm-poisson"
            )

    except Exception as exc:

        log.warning(
            "Impossible de lire models/meta.json: %s",
            exc,
        )

        MODEL_VERSION = (
            "lgbm-poisson"
        )

    log.info(
        "TikaML chargé: %s",
        MODEL_VERSION,
    )


@app.on_event("startup")
async def startup_event():

    try:

        load_predictor()

    except Exception as exc:

        log.exception(
            "Erreur chargement TikaML: %s",
            exc,
        )


# ============================================================
# OPENFOOTBALL
# ============================================================

def fetch_openfootball(
    league_code: str,
) -> dict[str, Any]:

    league_code = league_code.upper()

    if league_code not in LEAGUES:
        raise ValueError(
            f"Ligue inconnue: {league_code}"
        )

    now = time.time()

    cached = _openfootball_cache.get(
        league_code
    )

    if cached:

        if now - cached["timestamp"] < CACHE_TTL:

            return cached["data"]

    filename = LEAGUES[
        league_code
    ]["file"]

    url = (
        f"{OPENFOOTBALL_BASE}/{filename}"
    )

    log.info(
        "Téléchargement OpenFootball: %s",
        url,
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    _openfootball_cache[
        league_code
    ] = {
        "timestamp": now,
        "data": data,
    }

    return data


# ============================================================
# EXTRACTION DES MATCHS
# ============================================================

def extract_matches(
    data: dict[str, Any],
) -> list[dict[str, Any]]:

    matches: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # Format 1:
    # {
    #   "matches": [...]
    # }
    # --------------------------------------------------------

    if isinstance(
        data.get("matches"),
        list,
    ):

        matches.extend(
            data["matches"]
        )

    # --------------------------------------------------------
    # Format 2:
    # {
    #   "rounds": [
    #       {
    #          "matches": [...]
    #       }
    #   ]
    # }
    # --------------------------------------------------------

    rounds = data.get(
        "rounds",
        [],
    )

    if isinstance(
        rounds,
        list,
    ):

        for round_data in rounds:

            if not isinstance(
                round_data,
                dict,
            ):
                continue

            round_matches = round_data.get(
                "matches",
                [],
            )

            if not isinstance(
                round_matches,
                list,
            ):
                continue

            for match in round_matches:

                match_copy = dict(match)

                if "round" not in match_copy:

                    match_copy[
                        "round"
                    ] = round_data.get(
                        "name",
                        round_data.get(
                            "round"
                        ),
                    )

                matches.append(
                    match_copy
                )

    return matches


# ============================================================
# DATE MATCH
# ============================================================

def parse_match_date(
    match: dict[str, Any],
) -> Optional[date]:

    raw_date = match.get(
        "date"
    )

    if not raw_date:
        return None

    try:

        return datetime.strptime(
            str(raw_date)[:10],
            "%Y-%m-%d",
        ).date()

    except ValueError:

        return None


# ============================================================
# WEEK / JOURNEE
# ============================================================

def parse_week(
    match: dict[str, Any],
) -> Optional[int]:

    raw = str(
        match.get(
            "round",
            ""
        )
    )

    found = re.search(
        r"(\d+)",
        raw,
    )

    if not found:
        return None

    try:
        return int(
            found.group(1)
        )

    except ValueError:
        return None


# ============================================================
# RECHERCHE NOM TIKAML
# ============================================================

def resolve_tikaml_team(
    openfootball_name: str,
) -> Optional[str]:

    if not club_predictor:
        return None

    try:

        club_predictor.load_data()

        df = club_predictor.df

        if df is None:
            return None

        teams = set(
            df["home_team"]
            .dropna()
            .astype(str)
            .tolist()
        )

        teams.update(
            df["away_team"]
            .dropna()
            .astype(str)
            .tolist()
        )

        normalized_map = {
            normalize_team_name(team): team
            for team in teams
        }

        candidates = TEAM_ALIASES.get(
            openfootball_name,
            [openfootball_name],
        )

        # Nom direct
        candidates = [
            openfootball_name
        ] + candidates

        for candidate in candidates:

            key = normalize_team_name(
                candidate
            )

            if key in normalized_map:

                return normalized_map[key]

        # Comparaison souple
        source = normalize_team_name(
            openfootball_name
        )

        for key, original in normalized_map.items():

            if source == key:
                return original

            if (
                source in key
                or key in source
            ):

                return original

        return None

    except Exception as exc:

        log.warning(
            "Erreur résolution équipe %s: %s",
            openfootball_name,
            exc,
        )

        return None


# ============================================================
# FORMATAGE MATCH
# ============================================================

def format_openfootball_match(
    match: dict[str, Any],
    league_code: str,
) -> dict[str, Any]:

    return {
        "league": league_code,

        "league_name": LEAGUES[
            league_code
        ]["name"],

        "country": LEAGUES[
            league_code
        ]["country"],

        "round": match.get(
            "round"
        ),

        "date": match.get(
            "date"
        ),

        "time": match.get(
            "time"
        ),

        "home_team": match.get(
            "team1",
            match.get(
                "homeTeam",
                ""
            ),
        ),

        "away_team": match.get(
            "team2",
            match.get(
                "awayTeam",
                ""
            ),
        ),

        "score": match.get(
            "score"
        ),
    }


# ============================================================
# MATCHS D'UNE DATE
# ============================================================

def get_matches_for_date(
    target_date: date,
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
]:

    matches = []

    source_errors = {}

    for league_code in LEAGUES:

        try:

            data = fetch_openfootball(
                league_code
            )

            raw_matches = extract_matches(
                data
            )

            for raw_match in raw_matches:

                match_date = parse_match_date(
                    raw_match
                )

                if match_date != target_date:
                    continue

                matches.append(
                    format_openfootball_match(
                        raw_match,
                        league_code,
                    )
                )

        except Exception as exc:

            source_errors[
                league_code
            ] = str(exc)

    return (
        matches,
        source_errors,
    )


# ============================================================
# SCORE MATRIX → TOP SCORES
# ============================================================

def score_matrix_to_top_scores(
    matrix: Any,
    max_items: int = 5,
) -> list[dict[str, Any]]:

    if matrix is None:
        return []

    try:

        arr = np.asarray(
            matrix,
            dtype=float,
        )

        if arr.ndim != 2:
            return []

        items = []

        for i in range(
            arr.shape[0]
        ):

            for j in range(
                arr.shape[1]
            ):

                probability = float(
                    arr[i, j]
                )

                items.append(
                    {
                        "home_goals": i,
                        "away_goals": j,
                        "score": f"{i}-{j}",
                        "probability": round(
                            probability,
                            4,
                        ),
                    }
                )

        items.sort(
            key=lambda x: x[
                "probability"
            ],
            reverse=True,
        )

        return items[:max_items]

    except Exception:

        return []


# ============================================================
# TOP SCORES TIKAML
# ============================================================

def normalize_top_scores(
    result: dict[str, Any],
) -> list[dict[str, Any]]:

    raw = result.get(
        "top_scores",
        [],
    )

    if not raw:
        raw = result.get(
            "score_predictions",
            [],
        )

    normalized = []

    for item in raw:

        # tuple/list:
        # (home, away, probability)
        if isinstance(
            item,
            (tuple, list),
        ):

            if len(item) >= 3:

                try:

                    home_goals = int(
                        item[0]
                    )

                    away_goals = int(
                        item[1]
                    )

                    probability = normalize_probability(
                        item[2]
                    )

                    normalized.append(
                        {
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "score": (
                                f"{home_goals}-"
                                f"{away_goals}"
                            ),
                            "probability": round(
                                probability,
                                4,
                            ),
                        }
                    )

                except Exception:
                    pass

            continue

        # dict
        if isinstance(
            item,
            dict,
        ):

            home_goals = item.get(
                "home_goals",
                item.get(
                    "home",
                    0,
                ),
            )

            away_goals = item.get(
                "away_goals",
                item.get(
                    "away",
                    0,
                ),
            )

            probability = item.get(
                "probability",
                item.get(
                    "prob",
                    0,
                ),
            )

            try:

                home_goals = int(
                    home_goals
                )

                away_goals = int(
                    away_goals
                )

                probability = normalize_probability(
                    probability
                )

                normalized.append(
                    {
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "score": (
                            f"{home_goals}-"
                            f"{away_goals}"
                        ),
                        "probability": round(
                            probability,
                            4,
                        ),
                    }
                )

            except Exception:
                pass

    return normalized[:5]


# ============================================================
# CONVERSION RESULTAT TIKAML
# ============================================================

def convert_tikaml_prediction(
    result: dict[str, Any],
    home_team: str,
    away_team: str,
    league: str,
    match_date: str,
    kickoff: Optional[str] = None,
) -> dict[str, Any]:

    probs = result.get(
        "probs_1x2",
        [],
    )

    if len(probs) >= 3:

        home_probability = normalize_probability(
            probs[0]
        )

        draw_probability = normalize_probability(
            probs[1]
        )

        away_probability = normalize_probability(
            probs[2]
        )

    else:

        home_probability = normalize_probability(
            result.get(
                "home_probability",
                result.get(
                    "home_win",
                    0,
                ),
            )
        )

        draw_probability = normalize_probability(
            result.get(
                "draw_probability",
                result.get(
                    "draw",
                    0,
                ),
            )
        )

        away_probability = normalize_probability(
            result.get(
                "away_probability",
                result.get(
                    "away_win",
                    0,
                ),
            )
        )

    lambda_home = safe_float(
        result.get(
            "lambda_home",
            result.get(
                "expected_home",
                0,
            ),
        )
    )

    lambda_away = safe_float(
        result.get(
            "lambda_away",
            result.get(
                "expected_away",
                0,
            ),
        )
    )

    top_scores = normalize_top_scores(
        result
    )

    if not top_scores:

        matrix = result.get(
            "score_matrix"
        )

        top_scores = score_matrix_to_top_scores(
            matrix
        )

    recommended_score = result.get(
        "recommended_score"
    )

    if recommended_score:

        if isinstance(
            recommended_score,
            dict,
        ):

            score_label = recommended_score.get(
                "label"
            )

            score_probability = normalize_probability(
                recommended_score.get(
                    "prob",
                    0,
                )
            )

        else:

            score_label = str(
                recommended_score
            )

            score_probability = 0.0

    else:

        score_label = (
            top_scores[0]["score"]
            if top_scores
            else None
        )

        score_probability = (
            top_scores[0]["probability"]
            if top_scores
            else 0.0
        )

    corners_result = result.get(
        "corners",
        {},
    )

    if not corners_result:

        corners_result = {}

    yellows_result = result.get(
        "yellows",
        result.get(
            "cards",
            {},
        ),
    )

    if not yellows_result:

        yellows_result = {}

    corners = {
        "home": safe_float(
            corners_result.get(
                "expected_home",
                0,
            )
        ),
        "away": safe_float(
            corners_result.get(
                "expected_away",
                0,
            )
        ),
    }

    cards = {
        "home": safe_float(
            yellows_result.get(
                "expected_home",
                0,
            )
        ),
        "away": safe_float(
            yellows_result.get(
                "expected_away",
                0,
            )
        ),
    }

    selection_input = {
        "home_team": home_team,
        "away_team": away_team,

        "probabilities": {
            "home": home_probability,
            "draw": draw_probability,
            "away": away_probability,
        },

        "lambdas": {
            "home": lambda_home,
            "away": lambda_away,
        },

        "corners": corners,

        "cards": cards,

        "top_scores": top_scores,
    }

    # ========================================================
    # MOTEUR GAGNE TEMPS
    # ========================================================

    selection = select_prediction(
        selection_input
    )

    # ========================================================
    # SORTIE
    # ========================================================

    return {

        "league": league,

        "league_name": LEAGUES.get(
            league,
            {}
        ).get(
            "name",
            league,
        ),

        "date": match_date,

        "time": kickoff,

        "home_team": home_team,

        "away_team": away_team,

        "model": (
            "TikaML MatchPredictor"
        ),

        "version": MODEL_VERSION,

        "probabilities": {
            "home": percentage(
                home_probability
            ),
            "draw": percentage(
                draw_probability
            ),
            "away": percentage(
                away_probability
            ),
        },

        "lambdas": {
            "home": round(
                lambda_home,
                4,
            ),
            "away": round(
                lambda_away,
                4,
            ),
        },

        "recommended_score": {
            "score": score_label,
            "probability": percentage(
                score_probability
            ),
        },

        "top_scores": top_scores,

        "corners": corners,

        "cards": cards,

        "gagne_temps": selection,

        "raw_model": {
            "score_matrix": result.get(
                "score_matrix"
            ),
        },
    }


# ============================================================
# PREDICTION D'UN MATCH
# ============================================================

def predict_match(
    match: dict[str, Any],
) -> dict[str, Any]:

    if club_predictor is None:

        raise RuntimeError(
            "TikaML predictor not loaded"
        )

    openfootball_home = match[
        "home_team"
    ]

    openfootball_away = match[
        "away_team"
    ]

    league = match[
        "league"
    ]

    match_date = match[
        "date"
    ]

    kickoff = match.get(
        "time"
    )

    home_tika = resolve_tikaml_team(
        openfootball_home
    )

    away_tika = resolve_tikaml_team(
        openfootball_away
    )

    if not home_tika:

        raise ValueError(
            "Équipe introuvable dans TikaML: "
            f"{openfootball_home}"
        )

    if not away_tika:

        raise ValueError(
            "Équipe introuvable dans TikaML: "
            f"{openfootball_away}"
        )

    week = parse_week(
        match
    )

    normalized_season = current_season()

    log.info(
        "Prediction: %s vs %s | %s | %s",
        home_tika,
        away_tika,
        league,
        match_date,
    )

    # ========================================================
    # TIKAML
    # ========================================================

    result = club_predictor.predict(
        home_team=home_tika,
        away_team=away_tika,
        league=league,
        season=normalized_season,
        match_date=match_date,
        week=week,
        max_goals=MAX_GOALS,
        odds=None,
    )

    # ========================================================
    # CONVERSION + SELECTION ENGINE
    # ========================================================

    converted = convert_tikaml_prediction(
        result=result,
        home_team=openfootball_home,
        away_team=openfootball_away,
        league=league,
        match_date=match_date,
        kickoff=kickoff,
    )

    # Garder les noms TikaML pour diagnostic
    converted[
        "tikaml_teams"
    ] = {
        "home": home_tika,
        "away": away_tika,
    }

    return converted


# ============================================================
# SCORE DE CLASSEMENT GLOBAL
# ============================================================

def ranking_score(
    prediction: dict[str, Any],
) -> float:

    gagne = prediction.get(
        "gagne_temps",
        {},
    )

    global_selection = gagne.get(
        "global_selection",
        {},
    )

    level = global_selection.get(
        "level",
        "À ÉVITER",
    )

    confidence = safe_float(
        global_selection.get(
            "confidence",
            0,
        )
    )

    probability = normalize_probability(
        global_selection.get(
            "probability",
            0,
        )
    )

    level_weight = {
        "PREMIUM": 400,
        "BON PRONOSTIC": 300,
        "RISQUÉ": 200,
        "À ÉVITER": 0,
    }.get(
        level,
        0,
    )

    return (
        level_weight
        + confidence
        + probability * 100
    )


# ============================================================
# CLASSIFICATION TEXTE
# ============================================================

def selection_summary(
    prediction: dict[str, Any],
) -> dict[str, Any]:

    gagne = prediction.get(
        "gagne_temps",
        {},
    )

    selection = gagne.get(
        "global_selection",
        {},
    )

    return {
        "level": selection.get(
            "level",
            "À ÉVITER",
        ),
        "risk": selection.get(
            "risk",
            "TRÈS ÉLEVÉ",
        ),
        "market": selection.get(
            "market"
        ),
        "selection": selection.get(
            "selection"
        ),
        "probability": selection.get(
            "probability"
        ),
        "confidence": selection.get(
            "confidence"
        ),
    }


# ============================================================
# ROUTE RACINE
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": MODEL_VERSION,
        "predictor_loaded": (
            club_predictor is not None
        ),
        "source": (
            "OpenFootball + TikaML"
        ),
        "selection_engine": (
            "selection-engine-1.0"
        ),
        "leagues": list(
            LEAGUES.keys()
        ),
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


@app.get("/gagne-temps/health")
async def gagne_temps_health():

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": MODEL_VERSION,
        "predictor_loaded": (
            club_predictor is not None
        ),
        "openfootball": True,
        "selection_engine": True,
    }


# ============================================================
# LEAGUES
# ============================================================

@app.get("/gagne-temps/leagues")
async def gagne_temps_leagues():

    return {
        "status": "success",
        "leagues": [
            {
                "code": code,
                **info,
            }
            for code, info in LEAGUES.items()
        ],
    }


# ============================================================
# TODAY
# ============================================================

@app.get("/gagne-temps/today")
async def gagne_temps_today():

    today = datetime.now(
        timezone.utc
    ).date()

    matches, source_errors = (
        get_matches_for_date(
            today
        )
    )

    return {
        "status": "success",
        "source": "OpenFootball",
        "date": today.isoformat(),
        "count": len(matches),
        "matches": matches,
        "source_errors": source_errors,
    }


# ============================================================
# TOP
# ============================================================

@app.get("/gagne-temps/top")
async def gagne_temps_top():

    today = datetime.now(
        timezone.utc
    ).date()

    matches, source_errors = (
        get_matches_for_date(
            today
        )
    )

    predictions = []

    skipped = []

    for match in matches:

        try:

            prediction = predict_match(
                match
            )

            predictions.append(
                prediction
            )

        except Exception as exc:

            log.exception(
                "Match ignoré: %s vs %s",
                match.get(
                    "home_team"
                ),
                match.get(
                    "away_team"
                ),
            )

            skipped.append(
                {
                    "home_team": match.get(
                        "home_team"
                    ),
                    "away_team": match.get(
                        "away_team"
                    ),
                    "league": match.get(
                        "league"
                    ),
                    "reason": str(exc),
                }
            )

    # ========================================================
    # CLASSEMENT
    # ========================================================

    predictions.sort(
        key=ranking_score,
        reverse=True,
    )

    top_predictions = []

    for prediction in predictions:

        prediction[
            "selection_summary"
        ] = selection_summary(
            prediction
        )

        top_predictions.append(
            prediction
        )

    return {
        "status": "success",

        "source": (
            "OpenFootball + TikaML"
        ),

        "model": (
            "TikaML MatchPredictor"
        ),

        "selection_engine": (
            "GAGNE TEMPS Selection Engine"
        ),

        "version": MODEL_VERSION,

        "date": today.isoformat(),

        "count": len(
            top_predictions
        ),

        "top_5": top_predictions[:5],

        "all_predictions": top_predictions,

        "skipped": skipped,

        "source_errors": source_errors,
    }


# ============================================================
# PREDICT PUBLIC
# ============================================================

class PublicPredictionRequest(
    BaseModel
):

    league: str = Field(
        ...,
        description=(
            "EPL, LL, SEA, BUN ou LI1"
        ),
    )

    home_team: str

    away_team: str

    match_date: str

    time: Optional[str] = None


@app.post(
    "/gagne-temps/predict"
)
async def gagne_temps_predict(
    request: PublicPredictionRequest,
):

    league = request.league.upper()

    if league not in LEAGUES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Ligue invalide. "
                "Utilisez EPL, LL, SEA, BUN ou LI1."
            ),
        )

    match = {
        "league": league,
        "date": request.match_date,
        "time": request.time,
        "home_team": request.home_team,
        "away_team": request.away_team,
    }

    try:

        prediction = predict_match(
            match
        )

        prediction[
            "selection_summary"
        ] = selection_summary(
            prediction
        )

        return {
            "status": "success",
            "prediction": prediction,
        }

    except Exception as exc:

        log.exception(
            "Erreur prediction publique"
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# DEBUG OPENFOOTBALL
# ============================================================

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
            detail="Unknown league",
        )

    try:

        data = fetch_openfootball(
            league_code
        )

        matches = extract_matches(
            data
        )

        return {
            "status": "success",
            "league": league_code,
            "file": LEAGUES[
                league_code
            ]["file"],
            "count": len(
                matches
            ),
            "matches_sample": matches[
                :10
            ],
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# MODEL STATUS
# ============================================================

@app.get(
    "/model-status",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def model_status():

    predictor_loaded = (
        club_predictor is not None
    )

    goals_loaded = False
    corners_loaded = False
    yellows_loaded = False

    if club_predictor:

        goals_loaded = (
            club_predictor.model
            is not None
        )

        corners_loaded = (
            club_predictor.corner_model
            is not None
        )

        yellows_loaded = (
            club_predictor.yellow_model
            is not None
        )

    return {
        "status": "ok",

        "service": APP_NAME,

        "version": MODEL_VERSION,

        "predictor_loaded": (
            predictor_loaded
        ),

        "models_loaded": {
            "goals": goals_loaded,
            "corners": corners_loaded,
            "yellows": yellows_loaded,
        },

        "selection_engine": {
            "loaded": True,
            "version": (
                "selection-engine-1.0"
            ),
        },
    }


# ============================================================
# REQUEST TIKAML ORIGINAL
# ============================================================

class MatchContext(BaseModel):

    minute: Optional[int] = None

    second: int = 0

    period: Optional[str] = None

    status: Optional[str] = None

    home_score: int = 0

    away_score: int = 0

    home_team_id: Optional[str] = None

    away_team_id: Optional[str] = None

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

    match_context: Optional[
        MatchContext
    ] = None

    models: list[str] = Field(
        default_factory=lambda: [
            "goals",
            "corners",
            "yellows",
        ]
    )


# ============================================================
# ROUTE /predict
# ============================================================

@app.post(
    "/predict",
    dependencies=[
        Depends(verify_api_key)
    ],
)
async def original_predict(
    request: PredictionRequest,
):

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )

    # Cette route est conservée pour
    # compatibilité avec TikaML.
    #
    # GAGNE TEMPS utilise principalement
    # /gagne-temps/top et
    # /gagne-temps/predict.

    try:

        # Construction manuelle avec le modèle
        # TikaML sous-jacent.

        from src.lgbm_poisson import (
            LGBMPoissonModel,
        )

        model = club_predictor.model

        if model is None:

            raise RuntimeError(
                "Goal model not loaded"
            )

        feature_cols = (
            model.feature_cols
        )

        row = {}

        for column in feature_cols:

            value = request.feature_vector.get(
                column
            )

            row[column] = (
                float(value)
                if value is not None
                else np.nan
            )

        feature_df = pd.DataFrame(
            [row]
        )

        lh, la = (
            model.predict_lambdas(
                feature_df
            )
        )

        lh = float(lh[0])
        la = float(la[0])

        matrix = (
            model.predict_score_matrix(
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
            np.trace(
                matrix
            )
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

        best_i, best_j = divmod(
            int(
                np.argmax(matrix)
            ),
            MAX_GOALS,
        )

        top_scores = (
            score_matrix_to_top_scores(
                matrix
            )
        )

        return {
            "status": "success",

            "predictions": {
                "goals": {
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

                    "recommended_score": {
                        "home_goals": best_i,
                        "away_goals": best_j,
                        "prob": round(
                            float(
                                matrix[
                                    best_i,
                                    best_j,
                                ]
                            ),
                            4,
                        ),
                        "label": (
                            f"{best_i}-{best_j}"
                        ),
                    },

                    "top_scores": top_scores,
                }
            },

            "model_metadata": {
                "version": MODEL_VERSION,
                "prediction_type": (
                    request.prediction_type
                ),
            },
        }

    except Exception as exc:

        log.exception(
            "Erreur /predict"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
