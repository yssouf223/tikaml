"""
GAGNE TEMPS + TikaML Prediction API

Architecture:
- TikaML pour les prédictions
- OpenFootball pour les matchs actuels
- GAGNE TEMPS API pour l'application mobile
- API key conservée uniquement côté serveur
- Routes GAGNE TEMPS publiques pour l'application mobile
- Routes internes TikaML protégées par X-API-Key

Version corrigée:
- Suppression de l'appel invalide LGBMPoissonModel.predict()
- Utilisation de MatchPredictor.predict()
- MatchPredictor utilise predict_lambdas() + predict_score_matrix()
- Gestion robuste des erreurs
- OpenFootball 2026-27
"""

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

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.lgbm_poisson import (
    LGBMPoissonModel,
    FEATURE_COLS,
    CORNER_FEATURE_COLS,
    YELLOW_FEATURE_COLS,
)
from src.inference import MatchPredictor
from src.live_predictor import LivePredictor
from src import national_api


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("gagne-temps")

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = Path("models/corners")
YELLOW_MODEL_DIR = Path("models/yellows")
BACKFILL_DIR = Path("models/backfill_20260131")

MAX_GOALS = 7

OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/master"
)

OPENFOOTBALL_SEASON = "2026-27"

OPENFOOTBALL_LEAGUES = {
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

API_KEY = os.environ.get("TIKA_API_KEY", "")

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(
        "TIKA_API_KEY n'est pas configuree. "
        "Une cle temporaire a ete generee."
    )

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str = Security(api_key_header),
):
    """
    Protection des routes internes TikaML.

    IMPORTANT:
    La cle ne doit jamais être placée dans l'application mobile.
    """

    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ============================================================
# MODELES
# ============================================================

class Models:
    goals: LGBMPoissonModel | None = None
    corners: LGBMPoissonModel | None = None
    yellows: LGBMPoissonModel | None = None
    version: str = "unknown"


models = Models()
backfill_models = Models()

club_predictor: MatchPredictor | None = None
backfill_predictor: MatchPredictor | None = None


# ============================================================
# CACHE OPENFOOTBALL
# ============================================================

OPENFOOTBALL_CACHE: dict[str, dict] = {}

CACHE_TTL_SECONDS = 15 * 60


# ============================================================
# ALIASES EQUIPES
# ============================================================

TEAM_ALIASES = {
    # Bundesliga
    "1. FC Union Berlin": [
        "Union Berlin",
        "1. FC Union Berlin",
    ],
    "FC Schalke 04": [
        "Schalke 04",
        "FC Schalke 04",
    ],

    # Ligue 1
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

    # La Liga
    "Sevilla FC": [
        "Sevilla",
        "Sevilla FC",
    ],
    "Valencia CF": [
        "Valencia",
        "Valencia CF",
    ],

    # Serie A
    "Venezia FC": [
        "Venezia",
        "Venezia FC",
    ],
    "ACF Fiorentina": [
        "Fiorentina",
        "ACF Fiorentina",
    ],
}


def normalize_team_name(name: str) -> str:
    """
    Normalise un nom d'équipe pour les comparaisons.
    """

    if not name:
        return ""

    value = str(name).strip().lower()

    value = re.sub(
        r"[^a-z0-9àâäçéèêëîïôöùûüÿñ .'-]",
        "",
        value,
    )

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def get_team_candidates(team_name: str) -> list[str]:
    """
    Retourne les différentes variantes connues.
    """

    candidates = [team_name]

    if team_name in TEAM_ALIASES:
        candidates.extend(TEAM_ALIASES[team_name])

    for canonical, aliases in TEAM_ALIASES.items():
        if team_name in aliases:
            candidates.append(canonical)
            candidates.extend(aliases)

    result = []

    seen = set()

    for value in candidates:
        norm = normalize_team_name(value)

        if norm and norm not in seen:
            seen.add(norm)
            result.append(value)

    return result


def resolve_team_for_tikaml(
    team_name: str,
    predictor: MatchPredictor,
) -> str:
    """
    Trouve le nom exact d'une équipe dans features.csv.
    """

    candidates = get_team_candidates(team_name)

    predictor.load_data()

    if predictor.df is None:
        raise ValueError("Historical feature data unavailable.")

    teams = set()

    if "home_team" in predictor.df.columns:
        teams.update(
            predictor.df["home_team"]
            .dropna()
            .astype(str)
            .tolist()
        )

    if "away_team" in predictor.df.columns:
        teams.update(
            predictor.df["away_team"]
            .dropna()
            .astype(str)
            .tolist()
        )

    normalized = {
        normalize_team_name(team): team
        for team in teams
    }

    # Exact normalized match
    for candidate in candidates:
        key = normalize_team_name(candidate)

        if key in normalized:
            return normalized[key]

    # Recherche partielle prudente
    requested = normalize_team_name(team_name)

    matches = [
        original
        for norm, original in normalized.items()
        if requested in norm or norm in requested
    ]

    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Team not found in TikaML dataset: {team_name}"
    )


# ============================================================
# OPENFOOTBALL
# ============================================================

def get_openfootball_url(league: str) -> str:
    if league not in OPENFOOTBALL_LEAGUES:
        raise ValueError(
            f"Unsupported league: {league}"
        )

    filename = OPENFOOTBALL_LEAGUES[league]["file"]

    return (
        f"{OPENFOOTBALL_BASE}/"
        f"{OPENFOOTBALL_SEASON}/"
        f"{filename}"
    )


def fetch_openfootball_league(
    league: str,
    force: bool = False,
) -> list[dict]:

    now = time.time()

    cached = OPENFOOTBALL_CACHE.get(league)

    if (
        cached
        and not force
        and now - cached["timestamp"] < CACHE_TTL_SECONDS
    ):
        return cached["matches"]

    url = get_openfootball_url(league)

    log.info(
        "OpenFootball: chargement %s",
        url,
    )

    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "GAGNE-TEMPS/1.0",
        },
    )

    response.raise_for_status()

    data = response.json()

    matches = data.get("matches", [])

    if not isinstance(matches, list):
        matches = []

    OPENFOOTBALL_CACHE[league] = {
        "timestamp": now,
        "matches": matches,
    }

    return matches


def fetch_all_openfootball_matches(
    force: bool = False,
) -> tuple[list[dict], dict]:

    all_matches = []
    source_errors = {}

    for league, config in OPENFOOTBALL_LEAGUES.items():

        try:
            matches = fetch_openfootball_league(
                league,
                force=force,
            )

            for match in matches:
                item = dict(match)

                item["_league"] = league
                item["_league_name"] = config["name"]
                item["_country"] = config["country"]

                all_matches.append(item)

        except Exception as exc:

            log.exception(
                "OpenFootball error %s",
                league,
            )

            source_errors[league] = str(exc)

    return all_matches, source_errors


# ============================================================
# DATE / SAISON
# ============================================================

def normalize_season(value: str | None) -> str:

    if not value:
        return "2026-2027"

    value = str(value).strip()

    # 2026-27 -> 2026-2027
    match = re.fullmatch(
        r"(\d{4})-(\d{2})",
        value,
    )

    if match:
        year1 = int(match.group(1))
        year2_short = int(match.group(2))

        century = (year1 // 100) * 100
        year2 = century + year2_short

        if year2 <= year1:
            year2 += 100

        return f"{year1}-{year2}"

    return value


def normalize_date(value: Any) -> str:

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    return str(value)[:10]


def get_match_date(
    match: dict,
) -> str | None:

    value = match.get("date")

    if not value:
        return None

    return normalize_date(value)


def get_round_number(
    match: dict,
) -> int | None:

    round_value = str(
        match.get("round", "")
    )

    found = re.search(
        r"(\d+)",
        round_value,
    )

    if not found:
        return None

    return int(found.group(1))


# ============================================================
# SCORE / PROBABILITES
# ============================================================

def clean_json(value: Any) -> Any:
    """
    Convertit numpy / pandas / NaN en JSON compatible.
    """

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
        return clean_json(
            value.tolist()
        )

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

        return value

    if pd.isna(value):
        return None

    return value


def probability_percent(
    value: float,
) -> float:
    return round(
        float(value) * 100,
        1,
    )


def get_best_side(
    probs: list[float],
) -> tuple[str, int]:

    labels = [
        "home",
        "draw",
        "away",
    ]

    index = int(
        np.argmax(probs)
    )

    return labels[index], index


def get_confidence(
    probs: list[float],
) -> dict:

    sorted_probs = sorted(
        [float(x) for x in probs],
        reverse=True,
    )

    best = sorted_probs[0]

    second = (
        sorted_probs[1]
        if len(sorted_probs) > 1
        else 0.0
    )

    margin = best - second

    # Score conservateur.
    confidence = (
        50.0
        + margin * 100.0
        + max(0.0, best - 0.33) * 30.0
    )

    confidence = max(
        0.0,
        min(
            confidence,
            95.0,
        ),
    )

    if confidence >= 75:
        level = "FORTE"
    elif confidence >= 60:
        level = "MOYENNE"
    else:
        level = "FAIBLE"

    return {
        "value": round(
            confidence,
            1,
        ),
        "level": level,
        "margin": round(
            margin,
            4,
        ),
    }


# ============================================================
# CONVERSION RESULTAT TIKAML
# ============================================================

def convert_tikaml_prediction(
    result: dict,
    openfootball_match: dict,
    league: str,
) -> dict:

    probs_raw = result.get(
        "probs_1x2",
        [0.0, 0.0, 0.0],
    )

    probs = [
        float(x)
        for x in probs_raw
    ]

    home_probability = probs[0]
    draw_probability = probs[1]
    away_probability = probs[2]

    side, side_index = get_best_side(
        probs
    )

    home_name = openfootball_match.get(
        "team1",
        "",
    )

    away_name = openfootball_match.get(
        "team2",
        "",
    )

    confidence = get_confidence(
        probs
    )

    recommended = result.get(
        "recommended_score",
        {},
    )

    score_label = recommended.get(
        "label",
    )

    if not score_label:

        home_goals = recommended.get(
            "home_goals",
            0,
        )

        away_goals = recommended.get(
            "away_goals",
            0,
        )

        score_label = (
            f"{home_goals}-{away_goals}"
        )

    kickoff = openfootball_match.get(
        "time"
    )

    league_info = OPENFOOTBALL_LEAGUES.get(
        league,
        {},
    )

    prediction = {
        "home_team": home_name,
        "away_team": away_name,

        "tika_home_team": None,
        "tika_away_team": None,

        "league": league,
        "league_name": league_info.get(
            "name",
            league,
        ),
        "country": league_info.get(
            "country",
            "",
        ),

        "date": get_match_date(
            openfootball_match
        ),

        "kickoff": kickoff,

        "round": openfootball_match.get(
            "round"
        ),

        "best_pick": (
            home_name
            if side == "home"
            else away_name
            if side == "away"
            else "Draw"
        ),

        "predicted_side": side,

        "probabilities": {
            "home": round(
                home_probability,
                4,
            ),
            "draw": round(
                draw_probability,
                4,
            ),
            "away": round(
                away_probability,
                4,
            ),
        },

        "home_probability": round(
            home_probability,
            4,
        ),

        "draw_probability": round(
            draw_probability,
            4,
        ),

        "away_probability": round(
            away_probability,
            4,
        ),

        "confidence": confidence["value"],
        "confidence_level": confidence["level"],
        "margin": confidence["margin"],

        "recommended_score": {
            "label": score_label,
            "home_goals": recommended.get(
                "home_goals"
            ),
            "away_goals": recommended.get(
                "away_goals"
            ),
            "prob": recommended.get(
                "prob"
            ),
        },

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

        "source": "OpenFootball + TikaML",
        "model": "TikaML MatchPredictor",
        "version": models.version,
    }

    return clean_json(
        prediction
    )


# ============================================================
# PREDICTION GAGNE TEMPS
# ============================================================

def predict_gagne_temps_match(
    openfootball_match: dict,
) -> dict:

    if club_predictor is None:
        raise RuntimeError(
            "TikaML MatchPredictor is not loaded."
        )

    league = openfootball_match.get(
        "_league"
    )

    home_open = openfootball_match.get(
        "team1",
        "",
    )

    away_open = openfootball_match.get(
        "team2",
        "",
    )

    match_date = get_match_date(
        openfootball_match
    )

    if not league:
        raise ValueError(
            "Missing league."
        )

    if not home_open or not away_open:
        raise ValueError(
            "Missing teams."
        )

    if not match_date:
        raise ValueError(
            "Missing match date."
        )

    home_tika = resolve_team_for_tikaml(
        home_open,
        club_predictor,
    )

    away_tika = resolve_team_for_tikaml(
        away_open,
        club_predictor,
    )

    week = get_round_number(
        openfootball_match
    )

    season = normalize_season(
        OPENFOOTBALL_SEASON
    )

    log.info(
        "Prediction: %s vs %s | %s | %s",
        home_tika,
        away_tika,
        league,
        match_date,
    )

    # ========================================================
    # IMPORTANT
    # ========================================================
    # MatchPredictor.predict() utilise en interne:
    #
    # predict_lambdas()
    # +
    # predict_score_matrix()
    #
    # et NON:
    #
    # LGBMPoissonModel.predict()
    #
    # C'est la correction principale du bug actuel.
    # ========================================================

    result = club_predictor.predict(
        home_team=home_tika,
        away_team=away_tika,
        league=league,
        season=season,
        match_date=match_date,
        week=week,
        max_goals=MAX_GOALS,
    )

    prediction = convert_tikaml_prediction(
        result=result,
        openfootball_match=openfootball_match,
        league=league,
    )

    prediction[
        "tika_home_team"
    ] = home_tika

    prediction[
        "tika_away_team"
    ] = away_tika

    return prediction


# ============================================================
# TOP PREDICTIONS
# ============================================================

def prediction_rank_score(
    prediction: dict,
) -> float:

    confidence = float(
        prediction.get(
            "confidence",
            0,
        )
    )

    margin = float(
        prediction.get(
            "margin",
            0,
        )
    )

    best_probability = max(
        float(
            prediction.get(
                "home_probability",
                0,
            )
        ),
        float(
            prediction.get(
                "draw_probability",
                0,
            )
        ),
        float(
            prediction.get(
                "away_probability",
                0,
            )
        ),
        ),
    )

    return (
        confidence * 0.60
        + margin * 100.0 * 0.25
        + best_probability * 100.0 * 0.15
    )


def get_predictions_for_date(
    target_date: str,
    force_refresh: bool = False,
) -> dict:

    matches, source_errors = (
        fetch_all_openfootball_matches(
            force=force_refresh
        )
    )

    predictions = []
    skipped = []

    for match in matches:

        league = match.get(
            "_league"
        )

        match_date = get_match_date(
            match
        )

        if match_date != target_date:
            continue

        # Match already finished:
        # GAGNE TEMPS only handles upcoming
        # predictions unless explicitly requested.
        if match.get("score", {}).get("ft"):
            continue

        home = match.get(
            "team1",
            "",
        )

        away = match.get(
            "team2",
            "",
        )

        try:

            prediction = (
                predict_gagne_temps_match(
                    match
                )
            )

            predictions.append(
                prediction
            )

        except Exception as exc:

            reason = str(exc)

            log.warning(
                "Match skipped: %s vs %s | %s",
                home,
                away,
                reason,
            )

            skipped.append(
                {
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "reason": reason,
                }
            )

    predictions.sort(
        key=prediction_rank_score,
        reverse=True,
    )

    return {
        "predictions": predictions,
        "skipped": skipped,
        "source_errors": source_errors,
    }


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


class GagneTempsPredictRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str | None = None
    match_date: str
    week: int | None = None

    odds: dict[str, float] | None = None


# ============================================================
# FEATURE DATAFRAME
# ============================================================

def _build_feature_df(
    feature_vector: dict,
    feature_list: list[str],
) -> pd.DataFrame:

    row = {}

    for col in feature_list:

        value = feature_vector.get(
            col
        )

        row[col] = (
            float(value)
            if value is not None
            else np.nan
        )

    return pd.DataFrame(
        [row]
    )


# ============================================================
# OVER / UNDER
# ============================================================

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
            "over": round(
                float(p_over),
                4,
            ),
            "under": round(
                float(1 - p_over),
                4,
            ),
        }

    return ou


def _poisson_over_under(
    lambda_total: float,
    lines: list[float],
) -> dict:

    from scipy.stats import poisson

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


def _live_over_under(
    lambda_remaining: float,
    current_total: int,
    lines: list[float],
) -> dict:

    from scipy.stats import poisson

    result = {}

    for line in lines:

        needed = (
            line
            - current_total
        )

        if needed <= 0:

            result[str(line)] = {
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


# ============================================================
# PREDICTION STANDARD TIKAML
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

        # CORRECTION PRINCIPALE:
        # LGBMPoissonModel ne possède PAS predict()
        # Il faut utiliser predict_lambdas().
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
                        best_j,
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
            "over_under": _goals_over_under(
                matrix
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
    # YELLOW CARDS
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
# LIVE PREDICTION
# ============================================================

def predict_live(
    feature_vector: dict,
    ctx: MatchContext,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    minute = (
        ctx.minute
        or 0
    )

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
            home_red_cards=(
                ctx.home_red_cards
            ),
            away_red_cards=(
                ctx.away_red_cards
            ),
        )

        live = (
            lp.get_probabilities()
        )

        rem = live[
            "remaining_matrix"
        ]

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

        recommended_score = {
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
        }

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
                        float(
                            v["over"]
                        ),
                        4,
                    ),
                    "under": round(
                        float(
                            v["under"]
                        ),
                        4,
                    ),
                }
                for k, v
                in live[
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
                for k, v
                in live[
                    "next_goal"
                ].items()
            },
            "score_matrix": score_matrix,
            "recommended_score": (
                recommended_score
            ),
        }

    r = max(
        0,
        (90 - minute) / 90,
    )

    # --------------------------------------------------------
    # CORNERS LIVE
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
            "lambda_remaining_home": round(
                clh * r,
                4,
            ),
            "lambda_remaining_away": round(
                cla * r,
                4,
            ),
            "over_under": (
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
                )
            ),
        }

    # --------------------------------------------------------
    # YELLOWS LIVE
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
            "lambda_remaining_home": round(
                ylh * r,
                4,
            ),
            "lambda_remaining_away": round(
                yla * r,
                4,
            ),
            "over_under": (
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
                )
            ),
        }

    return predictions


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global club_predictor
    global backfill_predictor

    log.info(
        "=================================================="
    )

    log.info(
        "Demarrage GAGNE TEMPS / TikaML"
    )

    log.info(
        "=================================================="
    )

    t0 = time.time()

    # --------------------------------------------------------
    # GOALS
    # --------------------------------------------------------

    try:

        models.goals = (
            LGBMPoissonModel.load(
                str(MODEL_DIR)
            )
        )

        log.info(
            "Goals model loaded: %s features",
            len(
                models.goals.feature_cols
            ),
        )

    except Exception as exc:

        log.exception(
            "Impossible de charger le modele goals: %s",
            exc,
        )

        raise

    # --------------------------------------------------------
    # CORNERS
    # --------------------------------------------------------

    if CORNER_MODEL_DIR.exists():

        try:

            models.corners = (
                LGBMPoissonModel.load(
                    str(
                        CORNER_MODEL_DIR
                    )
                )
            )

            log.info(
                "Corners model loaded"
            )

        except Exception as exc:

            log.warning(
                "Corners model unavailable: %s",
                exc,
            )

    # --------------------------------------------------------
    # YELLOWS
    # --------------------------------------------------------

    if YELLOW_MODEL_DIR.exists():

        try:

            models.yellows = (
                LGBMPoissonModel.load(
                    str(
                        YELLOW_MODEL_DIR
                    )
                )
            )

            log.info(
                "Yellows model loaded"
            )

        except Exception as exc:

            log.warning(
                "Yellows model unavailable: %s",
                exc,
            )

    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    meta_path = (
        MODEL_DIR
        / "meta.json"
    )

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

        except Exception as exc:

            log.warning(
                "Impossible de lire meta.json: %s",
                exc,
            )

    # --------------------------------------------------------
    # MATCH PREDICTOR
    # --------------------------------------------------------

    try:

        club_predictor = (
            MatchPredictor()
        )

        club_predictor.load_model()

        log.info(
            "MatchPredictor loaded"
        )

    except Exception as exc:

        log.exception(
            "MatchPredictor unavailable: %s",
            exc,
        )

        club_predictor = None

    # --------------------------------------------------------
    # BACKFILL
    # --------------------------------------------------------

    if BACKFILL_DIR.exists():

        bf_goals_dir = (
            BACKFILL_DIR
            / "goals"
        )

        bf_corners_dir = (
            BACKFILL_DIR
            / "corners"
        )

        bf_yellows_dir = (
            BACKFILL_DIR
            / "yellows"
        )

        try:

            if bf_goals_dir.exists():

                backfill_models.goals = (
                    LGBMPoissonModel.load(
                        str(
                            bf_goals_dir
                        )
                    )
                )

            if bf_corners_dir.exists():

                backfill_models.corners = (
                    LGBMPoissonModel.load(
                        str(
                            bf_corners_dir
                        )
                    )
                )

            if bf_yellows_dir.exists():

                backfill_models.yellows = (
                    LGBMPoissonModel.load(
                        str(
                            bf_yellows_dir
                        )
                    )
                )

            backfill_models.version = (
                "lgbm-poisson-backfill-20260131"
            )

            log.info(
                "Backfill models loaded"
            )

        except Exception as exc:

            log.warning(
                "Backfill unavailable: %s",
                exc,
            )

    # --------------------------------------------------------
    # BACKFILL PREDICTOR
    # --------------------------------------------------------

    try:

        backfill_features = (
            Path(
                "data/opta/processed/features.csv"
            )
        )

        if backfill_features.exists():

            backfill_predictor = (
                MatchPredictor(
                    features_path=(
                        backfill_features
                    )
                )
            )

            if backfill_models.goals:

                backfill_predictor.load_data()

                backfill_predictor.model = (
                    backfill_models.goals
                )

                backfill_predictor.corner_model = (
                    backfill_models.corners
                )

                backfill_predictor.yellow_model = (
                    backfill_models.yellows
                )

    except Exception as exc:

        log.warning(
            "Backfill predictor unavailable: %s",
            exc,
        )

    # --------------------------------------------------------
    # NATIONAL
    # --------------------------------------------------------

    try:

        nm = (
            national_api.load_national()
        )

        log.info(
            "National model loaded: %s teams",
            len(nm.attack),
        )

    except Exception as exc:

        log.warning(
            "National model not loaded: %s",
            exc,
        )

    log.info(
        "Tous les modeles charges en %.1fs",
        time.time() - t0,
    )

    log.info(
        "GAGNE TEMPS pret."
    )

    yield

    log.info(
        "Arret du serveur."
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="GAGNE TEMPS - TikaML API",
    version="2.0.0",
    description=(
        "Football prediction API powered by "
        "OpenFootball + TikaML"
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
# NATIONAL API PROTEGEE
# ============================================================

app.include_router(
    national_api.router,
    dependencies=[
        Depends(
            verify_api_key
        )
    ],
)


# ============================================================
# PUBLIC HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "GAGNE TEMPS",
        "version": models.version,
        "predictor_loaded": (
            club_predictor is not None
        ),
        "openfootball": True,
        "season": OPENFOOTBALL_SEASON,
    }


# ============================================================
# PUBLIC GAGNE TEMPS HEALTH
# ============================================================

@app.get(
    "/gagne-temps/health"
)
async def gagne_temps_health():

    return {
        "status": "ok",
        "service": "GAGNE TEMPS",
        "source": "OpenFootball + TikaML",
        "model": "TikaML MatchPredictor",
        "version": models.version,
        "predictor_loaded": (
            club_predictor is not None
        ),
        "season": OPENFOOTBALL_SEASON,
        "leagues": list(
            OPENFOOTBALL_LEAGUES.keys()
        ),
    }


# ============================================================
# PUBLIC LEAGUES
# ============================================================

@app.get(
    "/gagne-temps/leagues"
)
async def gagne_temps_leagues():

    return {
        "status": "success",
        "leagues": [
            {
                "code": code,
                **config,
            }
            for code, config
            in OPENFOOTBALL_LEAGUES.items()
        ],
    }


# ============================================================
# PUBLIC TODAY
# ============================================================

@app.get(
    "/gagne-temps/today"
)
async def gagne_temps_today(
    refresh: bool = False,
):

    target_date = (
        datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d")
    )

    data = (
        get_predictions_for_date(
            target_date,
            force_refresh=refresh,
        )
    )

    return clean_json(
        {
            "status": "success",
            "source": (
                "OpenFootball + TikaML"
            ),
            "model": (
                "TikaML MatchPredictor"
            ),
            "version": models.version,
            "date": target_date,
            "count": len(
                data["predictions"]
            ),
            "predictions": data[
                "predictions"
            ],
            "skipped": data[
                "skipped"
            ],
            "source_errors": data[
                "source_errors"
            ],
        }
    )


# ============================================================
# PUBLIC TOP
# ============================================================

@app.get(
    "/gagne-temps/top"
)
async def gagne_temps_top(
    refresh: bool = False,
    limit: int = 5,
):

    limit = max(
        1,
        min(
            int(limit),
            20,
        ),
    )

    target_date = (
        datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d")
    )

    data = (
        get_predictions_for_date(
            target_date,
            force_refresh=refresh,
        )
    )

    top_predictions = (
        data["predictions"][:limit]
    )

    return clean_json(
        {
            "status": "success",
            "source": (
                "OpenFootball + TikaML"
            ),
            "model": (
                "TikaML MatchPredictor"
            ),
            "version": models.version,
            "date": target_date,
            "count": len(
                top_predictions
            ),
            "top_5": top_predictions,
            "skipped": data[
                "skipped"
            ],
            "source_errors": data[
                "source_errors"
            ],
        }
    )


# ============================================================
# PUBLIC SINGLE MATCH PREDICTION
# ============================================================

@app.post(
    "/gagne-temps/predict"
)
async def gagne_temps_predict(
    req: GagneTempsPredictRequest,
):

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "TikaML MatchPredictor "
                "not loaded"
            ),
        )

    league = (
        req.league.upper().strip()
    )

    if league not in OPENFOOTBALL_LEAGUES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported league: {league}. "
                f"Supported: "
                f"{', '.join(OPENFOOTBALL_LEAGUES.keys())}"
            ),
        )

    try:

        home_tika = (
            resolve_team_for_tikaml(
                req.home_team,
                club_predictor,
            )
        )

        away_tika = (
            resolve_team_for_tikaml(
                req.away_team,
                club_predictor,
            )
        )

        season = normalize_season(
            req.season
        )

        result = (
            club_predictor.predict(
                home_team=home_tika,
                away_team=away_tika,
                league=league,
                season=season,
                match_date=req.match_date,
                week=req.week,
                max_goals=MAX_GOALS,
                odds=req.odds,
            )
        )

        fake_openfootball_match = {
            "team1": req.home_team,
            "team2": req.away_team,
            "date": req.match_date,
            "time": None,
            "round": (
                f"Matchday {req.week}"
                if req.week
                else None
            ),
        }

        prediction = (
            convert_tikaml_prediction(
                result=result,
                openfootball_match=(
                    fake_openfootball_match
                ),
                league=league,
            )
        )

        prediction[
            "tika_home_team"
        ] = home_tika

        prediction[
            "tika_away_team"
        ] = away_tika

        return clean_json(
            {
                "status": "success",
                "source": (
                    "OpenFootball + TikaML"
                ),
                "model": (
                    "TikaML MatchPredictor"
                ),
                "version": models.version,
                "prediction": prediction,
            }
        )

    except Exception as exc:

        log.exception(
            "GAGNE TEMPS prediction error"
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )


# ============================================================
# PROTECTED STANDARD TIKAML /predict
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[
        Depends(
            verify_api_key
        )
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

    try:

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

    except Exception as exc:

        log.exception(
            "Prediction error"
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    elapsed = round(
        (
            time.time()
            - t0
        ) * 1000,
        1,
    )

    log.info(
        "predict match_id=%s type=%s "
        "trigger=%s models=%s elapsed=%sms",
        req.match_id,
        req.prediction_type,
        req.trigger,
        list(
            predictions.keys()
        ),
        elapsed,
    )

    return PredictionResponse(
        predictions=clean_json(
            predictions
        ),
        model_metadata={
            "version": models.version,
            "prediction_type": (
                req.prediction_type
            ),
            "elapsed_ms": elapsed,
        },
    )


# ============================================================
# PROTECTED BACKFILL
# ============================================================

@app.post(
    "/backfill",
    response_model=PredictionResponse,
    dependencies=[
        Depends(
            verify_api_key
        )
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

    try:

        predictions = predict_prematch(
            req.feature_vector,
            req.models,
            m=backfill_models,
        )

    except Exception as exc:

        log.exception(
            "Backfill error"
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    elapsed = round(
        (
            time.time()
            - t0
        ) * 1000,
        1,
    )

    return PredictionResponse(
        predictions=clean_json(
            predictions
        ),
        model_metadata={
            "version": (
                backfill_models.version
            ),
            "prediction_type": "prematch",
            "elapsed_ms": elapsed,
        },
    )


# ============================================================
# PROTECTED MODEL STATUS
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

        "match_predictor": {
            "loaded": (
                club_predictor
                is not None
            ),
        },

        "backfill": {
            "loaded": (
                backfill_models.goals
                is not None
            ),
            "version": (
                backfill_models.version
            ),
        },

        "national": {
            "loaded": (
                national_api.state.model
                is not None
            ),
            "teams": (
                len(
                    national_api.state.model.attack
                )
                if national_api.state.model
                else 0
            ),
        },
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "service": "GAGNE TEMPS",
        "status": "online",
        "version": models.version,
        "source": "OpenFootball + TikaML",
        "season": OPENFOOTBALL_SEASON,

        "public_endpoints": [
            "/health",
            "/gagne-temps/health",
            "/gagne-temps/leagues",
            "/gagne-temps/today",
            "/gagne-temps/top",
            "/gagne-temps/predict",
        ],

        "protected_endpoints": [
            "/predict",
            "/backfill",
            "/model-status",
            "/national/*",
        ],
    }
