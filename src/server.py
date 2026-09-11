"""
GAGNE TEMPS API
Football prediction API using:
- OpenFootball
- TikaML MatchPredictor
- GAGNE TEMPS Selection Engine

Routes publiques:
    GET /
    GET /health
    GET /gagne-temps/health
    GET /gagne-temps/leagues
    GET /gagne-temps/today
    GET /gagne-temps/top
    GET /gagne-temps/predict

Routes protégées:
    GET /predict
    GET /model-status
    GET /debug/openfootball/{league_code}
"""

import os
import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.inference import MatchPredictor

try:
    from src.selection_engine import select_prediction
except Exception:
    select_prediction = None


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "GAGNE TEMPS"
MODEL_VERSION = "lgbm-poisson-85f"

MAX_GOALS = 7
OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/master"
)

REQUEST_TIMEOUT = 12

TIKA_API_KEY = os.getenv("TIKA_API_KEY", "")

CACHE_DIR = Path("/tmp/gagne_temps_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("gagne-temps")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="Football prediction API powered by TikaML + OpenFootball",
    version=MODEL_VERSION,
)


# ============================================================
# TIKAML PREDICTOR
# ============================================================

club_predictor: Optional[MatchPredictor] = None
predictor_error: Optional[str] = None


def load_predictor() -> Optional[MatchPredictor]:
    """
    Charge le modèle TikaML une seule fois.
    """
    global club_predictor
    global predictor_error

    if club_predictor is not None:
        return club_predictor

    try:
        log.info("Chargement du modèle TikaML...")

        predictor = MatchPredictor()

        # Important:
        # on utilise load_model(), pas model.predict()
        predictor.load_model()

        club_predictor = predictor
        predictor_error = None

        log.info("TikaML chargé avec succès.")

        return club_predictor

    except Exception as exc:
        predictor_error = str(exc)

        log.exception(
            "Impossible de charger le modèle TikaML: %s",
            exc,
        )

        return None


# Chargement au démarrage.
# On ne fait pas échouer complètement FastAPI si le modèle
# rencontre un problème.
@app.on_event("startup")
def startup_event():
    log.info("==========================================")
    log.info("Démarrage %s", APP_NAME)
    log.info("Version: %s", MODEL_VERSION)
    log.info("==========================================")

    load_predictor()


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(value: Any) -> Any:
    """
    Convertit récursivement:
    - numpy.ndarray
    - numpy.float32/64
    - numpy.int64
    - tuples
    - NaN
    - Infinity
    - dicts
    - listes

    en objets JSON compatibles.
    """

    if value is None:
        return None

    # bool/int/float/string natifs
    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # NumPy sans importer numpy explicitement
    # pour garder le serveur plus léger.
    module_name = getattr(type(value), "__module__", "")

    if module_name.startswith("numpy"):
        # ndarray
        if hasattr(value, "tolist"):
            try:
                return json_safe(value.tolist())
            except Exception:
                pass

        # numpy scalar
        if hasattr(value, "item"):
            try:
                return json_safe(value.item())
            except Exception:
                pass

    if isinstance(value, dict):
        result = {}

        for key, item in value.items():
            result[str(key)] = json_safe(item)

        return result

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    # Pandas Timestamp / datetime
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    # Fallback
    try:
        return str(value)
    except Exception:
        return None


# ============================================================
# OPENFOOTBALL
# ============================================================

LEAGUES = {
    "EPL": {
        "name": "Premier League",
        "file": "en.1.json",
    },
    "LL": {
        "name": "La Liga",
        "file": "es.1.json",
    },
    "SEA": {
        "name": "Serie A",
        "file": "it.1.json",
    },
    "BUN": {
        "name": "Bundesliga",
        "file": "de.1.json",
    },
    "LI1": {
        "name": "Ligue 1",
        "file": "fr.1.json",
    },
}


TEAM_ALIASES = {
    # Allemagne
    "1. FC Union Berlin": "Union Berlin",
    "FC Union Berlin": "Union Berlin",

    "FC Schalke 04": "Schalke 04",

    # France
    "Stade Rennais FC 1901": "Rennes",
    "Stade Rennais": "Rennes",

    "Olympique de Marseille": "Olympique Marseille",
    "Olympique Marseille": "Olympique Marseille",

    # Espagne
    "Sevilla FC": "Sevilla",
    "Sevilla": "Sevilla",

    "Valencia CF": "Valencia",
    "Valencia": "Valencia",

    # Italie
    "Venezia FC": "Venezia",
    "Venezia": "Venezia",

    "ACF Fiorentina": "Fiorentina",
    "Fiorentina": "Fiorentina",
}


def normalize_team_name(name: str) -> str:
    """
    Normalisation simple.
    """

    if not name:
        return ""

    name = " ".join(str(name).strip().split())

    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]

    return name


def candidate_team_names(name: str) -> List[str]:
    """
    Retourne plusieurs variantes possibles.
    """

    original = " ".join(str(name).strip().split())

    candidates = [
        original,
        normalize_team_name(original),
    ]

    # Ajoute les alias inverses
    for source, target in TEAM_ALIASES.items():
        if target == original:
            candidates.append(source)

    # Supprime doublons
    result = []

    for item in candidates:
        if item and item not in result:
            result.append(item)

    return result


def get_current_season() -> str:
    """
    OpenFootball utilise ici la saison 2026-27.
    Pour septembre 2026, nous sommes dans 2026-27.
    """

    now = datetime.now(timezone.utc)

    if now.month >= 7:
        start = now.year
        end = now.year + 1
    else:
        start = now.year - 1
        end = now.year

    return f"{start}-{str(end)[-2:]}"


def get_season_folder() -> str:
    """
    Format OpenFootball:
    2026-27
    """

    return get_current_season()


def openfootball_url(league_code: str) -> str:
    league = LEAGUES.get(league_code)

    if not league:
        raise ValueError(
            f"Ligue OpenFootball inconnue: {league_code}"
        )

    season = get_season_folder()

    return (
        f"{OPENFOOTBALL_BASE}/"
        f"{season}/"
        f"{league['file']}"
    )


def fetch_openfootball(league_code: str) -> Dict[str, Any]:
    """
    Télécharge un fichier OpenFootball.
    """

    url = openfootball_url(league_code)

    log.info(
        "OpenFootball GET %s",
        url,
    )

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "GAGNE-TEMPS/1.0",
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    data = response.json()

    return data


def extract_matches(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    OpenFootball peut utiliser:
        {"matches": [...]}

    ou:
        {"rounds": [{"matches": [...]}]}
    """

    matches = []

    # Format direct
    if isinstance(data, dict):
        direct = data.get("matches")

        if isinstance(direct, list):
            matches.extend(direct)

    # Format rounds
    if isinstance(data, dict):
        rounds = data.get("rounds")

        if isinstance(rounds, list):

            for round_data in rounds:

                if not isinstance(round_data, dict):
                    continue

                round_matches = round_data.get("matches", [])

                if isinstance(round_matches, list):

                    for match in round_matches:

                        if not isinstance(match, dict):
                            continue

                        # Conserve le numéro de journée
                        if "round" not in match:
                            match["round"] = round_data.get(
                                "name",
                                round_data.get("round"),
                            )

                        matches.append(match)

    return matches


def parse_round_number(value: Any) -> Optional[int]:
    """
    Exemples:
    Matchday 4 -> 4
    4 -> 4
    MD4 -> 4
    """

    if value is None:
        return None

    text = str(value).strip()

    digits = ""

    for char in text:
        if char.isdigit():
            digits += char

    if not digits:
        return None

    try:
        return int(digits)
    except Exception:
        return None


def convert_openfootball_match(
    match: Dict[str, Any],
    league_code: str,
) -> Optional[Dict[str, Any]]:

    home = (
        match.get("team1")
        or match.get("home")
        or match.get("home_team")
    )

    away = (
        match.get("team2")
        or match.get("away")
        or match.get("away_team")
    )

    date = match.get("date")

    if not home or not away or not date:
        return None

    score = match.get("score")

    result = {
        "home_team": str(home),
        "away_team": str(away),
        "home_team_tika": normalize_team_name(str(home)),
        "away_team_tika": normalize_team_name(str(away)),
        "league": league_code,
        "league_name": LEAGUES[league_code]["name"],
        "date": str(date),
        "time": match.get("time"),
        "round": match.get("round"),
        "week": parse_round_number(match.get("round")),
        "finished": bool(
            isinstance(score, dict)
            and score.get("ft") is not None
        ),
        "score": score,
    }

    return result


def get_matches_for_date(
    target_date,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:

    all_matches = []
    source_errors = {}

    date_string = target_date.isoformat()

    for league_code in LEAGUES:

        try:

            data = fetch_openfootball(league_code)

            raw_matches = extract_matches(data)

            log.info(
                "%s: %d matchs trouvés dans OpenFootball",
                league_code,
                len(raw_matches),
            )

            for raw in raw_matches:

                match = convert_openfootball_match(
                    raw,
                    league_code,
                )

                if not match:
                    continue

                if match["date"] != date_string:
                    continue

                all_matches.append(match)

        except Exception as exc:

            source_errors[league_code] = str(exc)

            log.exception(
                "Erreur OpenFootball %s",
                league_code,
            )

    # Tri par heure
    all_matches.sort(
        key=lambda x: (
            x.get("time") or "99:99",
            x.get("league") or "",
        )
    )

    return all_matches, source_errors


# ============================================================
# TEAM RESOLUTION
# ============================================================

def resolve_tika_team(
    predictor: MatchPredictor,
    requested_name: str,
) -> Optional[str]:

    candidates = candidate_team_names(requested_name)

    try:
        df = predictor.df

        if df is None:
            predictor.load_data()

        df = predictor.df

        if df is None:
            return None

        teams = set()

        if "home_team" in df.columns:
            teams.update(
                str(x)
                for x in df["home_team"].dropna().unique()
            )

        if "away_team" in df.columns:
            teams.update(
                str(x)
                for x in df["away_team"].dropna().unique()
            )

        # Exact
        for candidate in candidates:

            if candidate in teams:
                return candidate

        # Normalized comparison
        def normalize_for_compare(value: str) -> str:
            return (
                value.lower()
                .replace(".", "")
                .replace("-", " ")
                .replace("_", " ")
                .strip()
            )

        normalized_teams = {
            normalize_for_compare(team): team
            for team in teams
        }

        for candidate in candidates:

            key = normalize_for_compare(candidate)

            if key in normalized_teams:
                return normalized_teams[key]

        # Recherche partielle prudente
        for candidate in candidates:

            key = normalize_for_compare(candidate)

            for normalized, original in normalized_teams.items():

                if (
                    key == normalized
                    or key in normalized
                    or normalized in key
                ):
                    return original

    except Exception:
        log.exception(
            "Erreur résolution équipe %s",
            requested_name,
        )

    return None


# ============================================================
# TIKAML PREDICTION
# ============================================================

def predict_match(
    match: Dict[str, Any],
) -> Dict[str, Any]:

    predictor = load_predictor()

    if predictor is None:
        raise RuntimeError(
            predictor_error
            or "TikaML predictor unavailable"
        )

    home_open = match["home_team"]
    away_open = match["away_team"]

    home_tika = resolve_tika_team(
        predictor,
        home_open,
    )

    away_tika = resolve_tika_team(
        predictor,
        away_open,
    )

    if not home_tika:
        raise ValueError(
            f"Équipe introuvable dans TikaML: {home_open}"
        )

    if not away_tika:
        raise ValueError(
            f"Équipe introuvable dans TikaML: {away_open}"
        )

    league = match["league"]

    season = get_current_season()

    match_date = match["date"]

    week = match.get("week")

    log.info(
        "Prediction TikaML: %s vs %s | %s | %s",
        home_tika,
        away_tika,
        league,
        match_date,
    )

    result = predictor.predict(
        home_team=home_tika,
        away_team=away_tika,
        league=league,
        season=season,
        match_date=match_date,
        week=week,
        max_goals=MAX_GOALS,
        odds=None,
    )

    converted = convert_tikaml_prediction(
        result,
        match,
        home_tika,
        away_tika,
    )

    # Sélection GAGNE TEMPS
    if select_prediction is not None:

        try:
            selection = select_prediction(converted)

            converted["gagne_temps"] = json_safe(
                selection
            )

        except Exception as exc:

            log.exception(
                "Erreur Selection Engine: %s",
                exc,
            )

            converted["gagne_temps"] = {
                "status": "unavailable",
                "error": str(exc),
            }

    else:

        converted["gagne_temps"] = {
            "status": "unavailable",
            "error": "Selection Engine non chargé",
        }

    return json_safe(converted)


# ============================================================
# CONVERSION TIKAML
# ============================================================

def normalize_top_scores(
    top_scores: Any,
) -> List[Dict[str, Any]]:

    result = []

    if not top_scores:
        return result

    for item in top_scores:

        try:

            if isinstance(item, dict):

                home_goals = item.get(
                    "home_goals",
                    item.get("home"),
                )

                away_goals = item.get(
                    "away_goals",
                    item.get("away"),
                )

                probability = item.get(
                    "probability",
                    item.get("prob", 0),
                )

            else:

                home_goals = item[0]
                away_goals = item[1]
                probability = item[2]

            result.append(
                {
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "score": f"{int(home_goals)}-{int(away_goals)}",
                    "probability": float(probability),
                }
            )

        except Exception:
            continue

    return result


def normalize_1x2(
    probs: Any,
) -> Dict[str, float]:

    try:

        home = float(probs[0])
        draw = float(probs[1])
        away = float(probs[2])

    except Exception:

        home = 0.0
        draw = 0.0
        away = 0.0

    total = home + draw + away

    if total > 0:

        home /= total
        draw /= total
        away /= total

    return {
        "home": round(home, 6),
        "draw": round(draw, 6),
        "away": round(away, 6),
    }


def convert_tikaml_prediction(
    result: Dict[str, Any],
    match: Dict[str, Any],
    home_tika: str,
    away_tika: str,
) -> Dict[str, Any]:

    probabilities = normalize_1x2(
        result.get("probs_1x2", [])
    )

    best_values = [
        (
            "home",
            probabilities["home"],
        ),
        (
            "draw",
            probabilities["draw"],
        ),
        (
            "away",
            probabilities["away"],
        ),
    ]

    best_pick, best_probability = max(
        best_values,
        key=lambda x: x[1],
    )

    labels = {
        "home": "1",
        "draw": "X",
        "away": "2",
    }

    outcomes = {
        "home": "Victoire domicile",
        "draw": "Match nul",
        "away": "Victoire extérieur",
    }

    recommended = result.get(
        "recommended_score"
    ) or {}

    top_scores = normalize_top_scores(
        result.get("top_scores")
    )

    # Score recommandé
    recommended_score = {
        "home_goals": recommended.get(
            "home_goals"
        ),
        "away_goals": recommended.get(
            "away_goals"
        ),
        "score": recommended.get(
            "label"
        ),
        "probability": recommended.get(
            "prob"
        ),
    }

    # Corners
    corners = result.get("corners")

    if isinstance(corners, dict):

        corners_output = {
            "lambda_home": corners.get(
                "lambda_home"
            ),
            "lambda_away": corners.get(
                "lambda_away"
            ),
            "total_lambda": (
                safe_float(corners.get("lambda_home"))
                + safe_float(corners.get("lambda_away"))
            ),
            "over_under": corners.get(
                "over_under",
                {},
            ),
        }

    else:
        corners_output = None

    # Cartons
    yellows = result.get("yellows")

    if isinstance(yellows, dict):

        cards_output = {
            "lambda_home": yellows.get(
                "lambda_home"
            ),
            "lambda_away": yellows.get(
                "lambda_away"
            ),
            "total_lambda": (
                safe_float(yellows.get("lambda_home"))
                + safe_float(yellows.get("lambda_away"))
            ),
            "over_under": yellows.get(
                "over_under",
                {},
            ),
        }

    else:
        cards_output = None

    output = {
        "match": {
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "home_team_tika": home_tika,
            "away_team_tika": away_tika,
            "league": match["league"],
            "league_name": match["league_name"],
            "date": match["date"],
            "time": match.get("time"),
            "round": match.get("round"),
            "week": match.get("week"),
        },

        "prediction": {
            "best_pick": best_pick,
            "best_pick_label": labels[best_pick],
            "outcome": outcomes[best_pick],
            "probability": round(
                best_probability,
                6,
            ),

            "probabilities": probabilities,

            "home_probability": probabilities[
                "home"
            ],

            "draw_probability": probabilities[
                "draw"
            ],

            "away_probability": probabilities[
                "away"
            ],
        },

        "recommended_score": recommended_score,

        "top_scores": top_scores,

        "goals": {
            "lambda_home": result.get(
                "lambda_home"
            ),
            "lambda_away": result.get(
                "lambda_away"
            ),
            "over_under": result.get(
                "goals_over_under",
                {},
            ),
        },

        "corners": corners_output,

        "cards": cards_output,

        "score_groups": result.get(
            "score_groups",
            [],
        ),

        "model": {
            "name": "TikaML MatchPredictor",
            "version": MODEL_VERSION,
            "source": "OpenFootball + TikaML",
        },

        # Le score_matrix est conservé mais converti
        # proprement en JSON.
        "raw_model": {
            "score_matrix": json_safe(
                result.get("score_matrix")
            ),
        },
    }

    return json_safe(output)


# ============================================================
# UTILITAIRES
# ============================================================

def safe_float(value: Any) -> float:

    try:

        if value is None:
            return 0.0

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return 0.0

        return value

    except Exception:
        return 0.0


def selection_summary(
    prediction: Dict[str, Any],
) -> Dict[str, Any]:

    gagne = prediction.get(
        "gagne_temps"
    )

    if not isinstance(gagne, dict):
        return {}

    # On essaie plusieurs structures possibles
    # afin de rester compatible avec le selection_engine.
    return {
        "classification": (
            gagne.get("classification")
            or gagne.get("category")
            or gagne.get("niveau")
        ),

        "best_market": (
            gagne.get("best_market")
            or gagne.get("market")
            or gagne.get("recommended_market")
        ),

        "confidence": (
            gagne.get("confidence")
            or gagne.get("confidence_index")
        ),

        "risk": (
            gagne.get("risk")
            or gagne.get("risk_level")
        ),
    }


def ranking_score(
    prediction: Dict[str, Any],
) -> float:

    """
    Classement robuste.

    On privilégie:
    1. classification
    2. confiance
    3. probabilité
    4. marge entre le meilleur choix et le second
    """

    classification = (
        prediction
        .get("gagne_temps", {})
        .get("classification")
    )

    classification_scores = {
        "PREMIUM": 1000,
        "BON PRONOSTIC": 700,
        "RISQUÉ": 400,
        "À ÉVITER": 0,
    }

    base = classification_scores.get(
        str(classification).upper()
        if classification
        else "",
        100,
    )

    prediction_data = prediction.get(
        "prediction",
        {},
    )

    probability = safe_float(
        prediction_data.get(
            "probability"
        )
    )

    probs = prediction_data.get(
        "probabilities",
        {},
    )

    values = [
        safe_float(probs.get("home")),
        safe_float(probs.get("draw")),
        safe_float(probs.get("away")),
    ]

    values.sort(reverse=True)

    margin = 0.0

    if len(values) >= 2:
        margin = max(
            0.0,
            values[0] - values[1],
        )

    confidence = safe_float(
        prediction
        .get("gagne_temps", {})
        .get("confidence")
    )

    return (
        base
        + probability * 100
        + margin * 100
        + confidence
    )


# ============================================================
# ROOT
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
        "source": "OpenFootball + TikaML",
        "routes": {
            "health": "/health",
            "today": "/gagne-temps/today",
            "top": "/gagne-temps/top",
            "predict": "/gagne-temps/predict",
            "leagues": "/gagne-temps/leagues",
        },
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    predictor = load_predictor()

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": MODEL_VERSION,
        "predictor_loaded": predictor is not None,
        "predictor_error": (
            predictor_error
            if predictor is None
            else None
        ),
    }


@app.get("/gagne-temps/health")
async def gagne_temps_health():

    predictor = load_predictor()

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": MODEL_VERSION,
        "predictor_loaded": predictor is not None,
        "selection_engine_loaded": (
            select_prediction is not None
        ),
        "openfootball": True,
        "predictor_error": (
            predictor_error
            if predictor is None
            else None
        ),
    }


# ============================================================
# LEAGUES
# ============================================================

@app.get("/gagne-temps/leagues")
async def gagne_temps_leagues():

    return {
        "status": "success",
        "count": len(LEAGUES),
        "leagues": [
            {
                "code": code,
                "name": data["name"],
                "file": data["file"],
            }
            for code, data in LEAGUES.items()
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
        get_matches_for_date(today)
    )

    return {
        "status": "success",
        "source": "OpenFootball",
        "date": today.isoformat(),
        "count": len(matches),
        "matches": json_safe(matches),
        "source_errors": json_safe(
            source_errors
        ),
    }


# ============================================================
# TOP
# ============================================================

@app.get("/gagne-temps/top")
async def gagne_temps_top():

    today = datetime.now(
        timezone.utc
    ).date()

    log.info(
        "=========================================="
    )

    log.info(
        "GAGNE TEMPS TOP - %s",
        today.isoformat(),
    )

    log.info(
        "=========================================="
    )

    # --------------------------------------------------------
    # Récupération OpenFootball
    # --------------------------------------------------------

    try:

        matches, source_errors = (
            get_matches_for_date(today)
        )

    except Exception as exc:

        log.exception(
            "Erreur globale OpenFootball"
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "source": "OpenFootball",
                "date": today.isoformat(),
                "count": 0,
                "top_5": [],
                "all_predictions": [],
                "skipped": [],
                "source_errors": {
                    "global": str(exc),
                },
            },
        )

    # --------------------------------------------------------
    # Analyse match par match
    # --------------------------------------------------------

    predictions = []
    skipped = []

    for match in matches:

        home = match.get(
            "home_team",
            "",
        )

        away = match.get(
            "away_team",
            "",
        )

        league = match.get(
            "league",
            "",
        )

        try:

            log.info(
                "Analyse: %s vs %s [%s]",
                home,
                away,
                league,
            )

            prediction = predict_match(
                match
            )

            prediction[
                "selection_summary"
            ] = selection_summary(
                prediction
            )

            internal_score = ranking_score(
                prediction
            )

            prediction[
                "_ranking_score"
            ] = internal_score

            predictions.append(
                prediction
            )

            log.info(
                "OK: %s vs %s | ranking=%.2f",
                home,
                away,
                internal_score,
            )

        except Exception as exc:

            log.exception(
                "Erreur prediction: %s vs %s",
                home,
                away,
            )

            skipped.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "reason": str(exc),
                    "error_type": type(
                        exc
                    ).__name__,
                }
            )

    # --------------------------------------------------------
    # Classement
    # --------------------------------------------------------

    predictions.sort(
        key=lambda item: safe_float(
            item.get(
                "_ranking_score",
                0,
            )
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Nettoyage du score interne
    # --------------------------------------------------------

    for prediction in predictions:

        prediction.pop(
            "_ranking_score",
            None,
        )

    # --------------------------------------------------------
    # Réponse finale
    # --------------------------------------------------------

    response = {
        "status": "success",
        "source": "OpenFootball + TikaML",
        "model": "TikaML MatchPredictor",
        "selection_engine": (
            "GAGNE TEMPS Selection Engine"
        ),
        "version": MODEL_VERSION,
        "date": today.isoformat(),
        "count": len(predictions),
        "top_5": predictions[:5],
        "all_predictions": predictions,
        "skipped": skipped,
        "source_errors": source_errors,
    }

    # Dernière sécurité JSON
    safe_response = json_safe(
        response
    )

    return JSONResponse(
        status_code=200,
        content=safe_response,
    )


# ============================================================
# PREDICT BY QUERY
# ============================================================

@app.get("/gagne-temps/predict")
async def gagne_temps_predict(
    home_team: str,
    away_team: str,
    league: str,
    date: Optional[str] = None,
):

    league = league.upper().strip()

    if league not in LEAGUES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Ligue inconnue: {league}. "
                f"Utilise: "
                f"{', '.join(LEAGUES.keys())}"
            ),
        )

    if not date:

        date = datetime.now(
            timezone.utc
        ).date().isoformat()

    match = {
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "league_name": LEAGUES[
            league
        ]["name"],
        "date": date,
        "time": None,
        "round": None,
        "week": None,
    }

    try:

        prediction = predict_match(
            match
        )

        return JSONResponse(
            status_code=200,
            content=json_safe(
                prediction
            ),
        )

    except Exception as exc:

        log.exception(
            "Erreur prediction manuelle"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# COMPATIBILITY /predict
# ============================================================

@app.get("/predict")
async def predict_compat(
    home_team: str,
    away_team: str,
    league: str,
    date: Optional[str] = None,
):

    return await gagne_temps_predict(
        home_team=home_team,
        away_team=away_team,
        league=league,
        date=date,
    )


# ============================================================
# MODEL STATUS
# ============================================================

@app.get("/model-status")
async def model_status():

    predictor = load_predictor()

    if predictor is None:

        return {
            "status": "error",
            "loaded": False,
            "error": predictor_error,
            "version": MODEL_VERSION,
        }

    return {
        "status": "ok",
        "loaded": True,
        "version": MODEL_VERSION,
        "model": (
            type(
                predictor.model
            ).__name__
            if predictor.model is not None
            else None
        ),
        "corner_model": (
            predictor.corner_model
            is not None
        ),
        "yellow_model": (
            predictor.yellow_model
            is not None
        ),
    }


# ============================================================
# DEBUG OPENFOOTBALL
# ============================================================

@app.get(
    "/debug/openfootball/{league_code}"
)
async def debug_openfootball(
    league_code: str,
):

    league_code = league_code.upper()

    if league_code not in LEAGUES:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Ligue inconnue: {league_code}"
            ),
        )

    try:

        url = openfootball_url(
            league_code
        )

        data = fetch_openfootball(
            league_code
        )

        matches = extract_matches(
            data
        )

        today = datetime.now(
            timezone.utc
        ).date().isoformat()

        today_matches = []

        for raw in matches:

            match = convert_openfootball_match(
                raw,
                league_code,
            )

            if (
                match
                and match["date"]
                == today
            ):
                today_matches.append(
                    match
                )

        return {
            "status": "success",
            "league": league_code,
            "url": url,
            "season": get_current_season(),
            "total_matches": len(matches),
            "today": today,
            "today_count": len(
                today_matches
            ),
            "today_matches": json_safe(
                today_matches
            ),
        }

    except Exception as exc:

        log.exception(
            "Debug OpenFootball erreur"
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "league": league_code,
                "error": str(exc),
                "error_type": type(
                    exc
                ).__name__,
            },
        )


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc: Exception,
):

    log.exception(
        "Erreur API non gérée: %s",
        exc,
    )

    return JSONResponse(
        status_code=200,
        content={
            "status": "error",
            "service": APP_NAME,
            "version": MODEL_VERSION,
            "error": str(exc),
            "error_type": type(
                exc
            ).__name__,
            "path": str(
                request.url.path
            ),
        },
    )
