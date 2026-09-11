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
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from scipy.stats import poisson

from src.lgbm_poisson import (
    LGBMPoissonModel,
    FEATURE_COLS,
    CORNER_FEATURE_COLS,
    YELLOW_FEATURE_COLS,
)
from src.live_predictor import LivePredictor
from src import national_api
from src.inference import MatchPredictor


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("gagne-temps")

MODEL_DIR = Path("models")
CORNER_MODEL_DIR = Path("models/corners")
YELLOW_MODEL_DIR = Path("models/yellows")
BACKFILL_DIR = Path("models/backfill_20260131")

MAX_GOALS = 7

API_KEY = os.environ.get("TIKA_API_KEY", "")

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning("No TIKA_API_KEY configured. A temporary API key was generated.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================
# OPENFOOTBALL
# ============================================================

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

OPENFOOTBALL_CACHE_TTL = 20 * 60

_openfootball_cache: dict[str, dict[str, Any]] = {}
_openfootball_errors: dict[str, str] = {}


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

club_predictor: MatchPredictor | None = None


# ============================================================
# API KEY
# ============================================================

async def verify_api_key(
    key: str = Security(api_key_header),
):
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global club_predictor

    log.info("Loading TikaML models...")

    try:
        models.goals = LGBMPoissonModel.load(str(MODEL_DIR))
        log.info("Goal model loaded.")
    except Exception as exc:
        log.exception("Could not load goal model: %s", exc)

    try:
        if CORNER_MODEL_DIR.exists():
            models.corners = LGBMPoissonModel.load(
                str(CORNER_MODEL_DIR)
            )
            log.info("Corner model loaded.")
    except Exception as exc:
        log.exception("Could not load corner model: %s", exc)

    try:
        if YELLOW_MODEL_DIR.exists():
            models.yellows = LGBMPoissonModel.load(
                str(YELLOW_MODEL_DIR)
            )
            log.info("Yellow-card model loaded.")
    except Exception as exc:
        log.exception("Could not load yellow model: %s", exc)

    # --------------------------------------------------------
    # Backfill models
    # --------------------------------------------------------

    if BACKFILL_DIR.exists():

        try:
            backfill_models.goals = LGBMPoissonModel.load(
                str(BACKFILL_DIR)
            )
        except Exception as exc:
            log.warning("Backfill goal model unavailable: %s", exc)

        try:
            if (BACKFILL_DIR / "corners").exists():
                backfill_models.corners = LGBMPoissonModel.load(
                    str(BACKFILL_DIR / "corners")
                )
        except Exception as exc:
            log.warning("Backfill corner model unavailable: %s", exc)

        try:
            if (BACKFILL_DIR / "yellows").exists():
                backfill_models.yellows = LGBMPoissonModel.load(
                    str(BACKFILL_DIR / "yellows")
                )
        except Exception as exc:
            log.warning("Backfill yellow model unavailable: %s", exc)

    # --------------------------------------------------------
    # Version
    # --------------------------------------------------------

    models.version = "lgbm-poisson-84f"

    # --------------------------------------------------------
    # Club predictor
    # --------------------------------------------------------

    try:
        club_predictor = MatchPredictor()
        club_predictor.load_data()
        club_predictor.load_model()

        log.info(
            "GAGNE TEMPS predictor loaded (%s historical matches)",
            len(club_predictor.df),
        )

    except Exception as exc:
        club_predictor = None
        log.exception(
            "Could not initialize GAGNE TEMPS predictor: %s",
            exc,
        )

    # --------------------------------------------------------
    # National model
    # --------------------------------------------------------

    try:
        nm = national_api.load_national()

        if nm is not None:
            log.info("National model loaded.")

    except Exception as exc:
        log.warning(
            "National model unavailable: %s",
            exc,
        )

    yield


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="GAGNE TEMPS Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    national_api.router,
    dependencies=[Depends(verify_api_key)],
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


class GagneTempsRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str = "2026-27"
    match_date: str
    week: int | None = None


# ============================================================
# GENERIC HELPERS
# ============================================================

def _clean_number(value: Any):
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    return value


def clean_json(value: Any):

    if isinstance(value, dict):
        return {
            str(k): clean_json(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [clean_json(v) for v in value]

    return _clean_number(value)


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(maximum, value),
    )


def normalize_team_name(name: str) -> str:

    if not name:
        return ""

    value = str(name).lower().strip()

    value = value.replace("’", "'")

    value = re.sub(
        r"[^a-z0-9à-ÿ]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalize_league_code(league: str) -> str:

    if not league:
        return ""

    value = normalize_team_name(league)

    aliases = {
        "epl": "EPL",
        "premier league": "EPL",
        "premiership": "EPL",

        "ll": "LL",
        "la liga": "LL",
        "laliga": "LL",
        "liga": "LL",

        "sea": "SEA",
        "serie a": "SEA",

        "bun": "BUN",
        "bundesliga": "BUN",

        "li1": "LI1",
        "ligue 1": "LI1",
        "ligue1": "LI1",
    }

    return aliases.get(
        value,
        league.upper().strip(),
    )


def normalize_season(season: str) -> str:

    if not season:
        return "2026-2027"

    value = str(season).strip()

    if re.fullmatch(r"\d{4}", value):
        year = int(value)
        return f"{year}-{year + 1}"

    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value[:5] + value[5:]

    if re.fullmatch(r"\d{4}/\d{2}", value):
        return value.replace("/", "-")

    if re.fullmatch(r"\d{4}-\d{4}", value):
        return value

    return value


# ============================================================
# OPENFOOTBALL FETCHER
# ============================================================

def fetch_openfootball(
    league_code: str,
) -> dict:

    league_code = normalize_league_code(
        league_code
    )

    if league_code not in OPENFOOTBALL_LEAGUES:
        raise ValueError(
            f"Unsupported league: {league_code}"
        )

    now = time.time()

    cached = _openfootball_cache.get(
        league_code
    )

    if cached:
        if now - cached["timestamp"] < OPENFOOTBALL_CACHE_TTL:
            return cached["data"]

    league = OPENFOOTBALL_LEAGUES[
        league_code
    ]

    url = (
        f"{OPENFOOTBALL_BASE}/"
        f"{OPENFOOTBALL_SEASON}/"
        f"{league['file']}"
    )

    try:
        import urllib.request

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "GAGNE-TEMPS/1.0 "
                    "(football prediction service)"
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=20,
        ) as response:

            raw = response.read()

        data = json.loads(
            raw.decode("utf-8")
        )

        _openfootball_cache[
            league_code
        ] = {
            "timestamp": now,
            "data": data,
        }

        _openfootball_errors.pop(
            league_code,
            None,
        )

        return data

    except Exception as exc:

        _openfootball_errors[
            league_code
        ] = str(exc)

        if cached:
            log.warning(
                "OpenFootball failed for %s; "
                "using cached data.",
                league_code,
            )

            return cached["data"]

        raise


# ============================================================
# OPENFOOTBALL SCORE PARSER
# ============================================================

def parse_score(score: Any):

    if not score:
        return None

    ft = score

    if isinstance(score, dict):
        ft = score.get("ft")

        if ft is None:
            ft = score.get("final")

    if isinstance(ft, dict):

        home = (
            ft.get("home")
            if ft.get("home") is not None
            else ft.get("1")
        )

        away = (
            ft.get("away")
            if ft.get("away") is not None
            else ft.get("2")
        )

        try:
            return int(home), int(away)
        except Exception:
            return None

    if isinstance(ft, (list, tuple)):

        if len(ft) >= 2:
            try:
                return int(ft[0]), int(ft[1])
            except Exception:
                return None

    if isinstance(ft, str):

        match = re.search(
            r"(\d+)\s*[-:]\s*(\d+)",
            ft,
        )

        if match:
            return (
                int(match.group(1)),
                int(match.group(2)),
            )

    return None


def match_date_value(match: dict) -> str:

    value = str(
        match.get("date", "")
    ).strip()

    if len(value) >= 10:
        return value[:10]

    return value


def match_datetime_value(match: dict) -> str:

    date = match_date_value(match)

    time_value = str(
        match.get("time", "")
    ).strip()

    return f"{date} {time_value}".strip()


# ============================================================
# OPENFOOTBALL MATCHES
# ============================================================

def get_matches(
    data: dict,
) -> list[dict]:

    matches = data.get(
        "matches",
        [],
    )

    if not isinstance(
        matches,
        list,
    ):
        return []

    return matches


def is_completed_match(
    match: dict,
) -> bool:

    return parse_score(
        match.get("score")
    ) is not None


# ============================================================
# CURRENT STANDINGS
# ============================================================

def build_standings(
    data: dict,
) -> dict[str, dict]:

    stats: dict[str, dict] = {}

    def ensure(team: str):

        if team not in stats:

            stats[team] = {
                "team": team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "points": 0,
                "home_played": 0,
                "home_points": 0,
                "away_played": 0,
                "away_points": 0,
            }

        return stats[team]

    for match in get_matches(data):

        score = parse_score(
            match.get("score")
        )

        if score is None:
            continue

        home = match.get("team1")
        away = match.get("team2")

        if not home or not away:
            continue

        home_goals, away_goals = score

        h = ensure(home)
        a = ensure(away)

        h["played"] += 1
        a["played"] += 1

        h["gf"] += home_goals
        h["ga"] += away_goals

        a["gf"] += away_goals
        a["ga"] += home_goals

        h["home_played"] += 1
        a["away_played"] += 1

        if home_goals > away_goals:

            h["wins"] += 1
            a["losses"] += 1

            h["points"] += 3
            h["home_points"] += 3

        elif home_goals < away_goals:

            a["wins"] += 1
            h["losses"] += 1

            a["points"] += 3
            a["away_points"] += 3

        else:

            h["draws"] += 1
            a["draws"] += 1

            h["points"] += 1
            a["points"] += 1

            h["home_points"] += 1
            a["away_points"] += 1

    table = list(
        stats.values()
    )

    for row in table:
        row["gd"] = (
            row["gf"] -
            row["ga"]
        )

    table.sort(
        key=lambda x: (
            x["points"],
            x["gd"],
            x["gf"],
            x["wins"],
        ),
        reverse=True,
    )

    for index, row in enumerate(
        table,
        start=1,
    ):
        row["rank"] = index

    return {
        row["team"]: row
        for row in table
    }


# ============================================================
# TEAM FORM
# ============================================================

def team_form(
    data: dict,
    team: str,
    limit: int = 5,
):

    matches = []

    for match in get_matches(data):

        score = parse_score(
            match.get("score")
        )

        if score is None:
            continue

        home = match.get("team1")
        away = match.get("team2")

        if (
            normalize_team_name(home)
            != normalize_team_name(team)
            and
            normalize_team_name(away)
            != normalize_team_name(team)
        ):
            continue

        matches.append(
            (
                match_date_value(match),
                match,
                score,
            )
        )

    matches.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    result = []

    for _, match, score in matches[:limit]:

        home = match.get("team1")
        away = match.get("team2")

        home_goals, away_goals = score

        is_home = (
            normalize_team_name(home)
            == normalize_team_name(team)
        )

        if is_home:

            if home_goals > away_goals:
                result.append("W")
            elif home_goals < away_goals:
                result.append("L")
            else:
                result.append("D")

        else:

            if away_goals > home_goals:
                result.append("W")
            elif away_goals < home_goals:
                result.append("L")
            else:
                result.append("D")

    return result


def form_string(
    form: list[str],
) -> str:

    return "".join(form)


def form_points(
    form: list[str],
) -> int:

    return sum(
        3 if x == "W"
        else 1 if x == "D"
        else 0
        for x in form
    )


# ============================================================
# H2H
# ============================================================

def build_h2h(
    data: dict,
    home_team: str,
    away_team: str,
    limit: int = 5,
):

    target_home = normalize_team_name(
        home_team
    )

    target_away = normalize_team_name(
        away_team
    )

    matches = []

    for match in get_matches(data):

        score = parse_score(
            match.get("score")
        )

        if score is None:
            continue

        team1 = normalize_team_name(
            match.get("team1", "")
        )

        team2 = normalize_team_name(
            match.get("team2", "")
        )

        same_pair = (
            team1 == target_home
            and team2 == target_away
        ) or (
            team1 == target_away
            and team2 == target_home
        )

        if not same_pair:
            continue

        matches.append(
            (
                match_date_value(match),
                match,
                score,
            )
        )

    matches.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    selected = matches[:limit]

    home_wins = 0
    draws = 0
    away_wins = 0
    goal_diff_home = 0

    history = []

    for _, match, score in selected:

        team1 = match.get("team1")
        team2 = match.get("team2")

        g1, g2 = score

        if normalize_team_name(team1) == target_home:

            goal_diff_home += (
                g1 - g2
            )

            if g1 > g2:
                home_wins += 1
            elif g1 < g2:
                away_wins += 1
            else:
                draws += 1

        else:

            goal_diff_home += (
                g2 - g1
            )

            if g2 > g1:
                home_wins += 1
            elif g2 < g1:
                away_wins += 1
            else:
                draws += 1

        history.append(
            {
                "date": match_date_value(match),
                "home": team1,
                "away": team2,
                "score": f"{g1}-{g2}",
            }
        )

    total = len(selected)

    return {
        "matches": total,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "home_win_pct": (
            home_wins / total
            if total
            else None
        ),
        "goal_diff_home": goal_diff_home,
        "history": history,
    }


# ============================================================
# TIKAML TEAM ALIAS RESOLUTION
# ============================================================

TEAM_ALIASES = {
    "1. FC Union Berlin": [
        "1. FC Union Berlin",
        "Union Berlin",
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


def get_tika_team_names():

    if club_predictor is None:
        return []

    df = getattr(
        club_predictor,
        "df",
        None,
    )

    if df is None:
        return []

    names = set()

    for column in [
        "home_team",
        "away_team",
        "home",
        "away",
        "team",
    ]:

        if column in df.columns:

            values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
            )

            names.update(values)

    return list(names)


def resolve_tika_team(
    openfootball_name: str,
):

    if not openfootball_name:
        return None

    available = get_tika_team_names()

    if not available:
        return openfootball_name

    normalized_available = {
        normalize_team_name(x): x
        for x in available
    }

    direct = normalized_available.get(
        normalize_team_name(
            openfootball_name
        )
    )

    if direct:
        return direct

    candidates = TEAM_ALIASES.get(
        openfootball_name,
        [openfootball_name],
    )

    for candidate in candidates:

        found = normalized_available.get(
            normalize_team_name(candidate)
        )

        if found:
            return found

    # Reverse alias search
    for key, aliases in TEAM_ALIASES.items():

        if normalize_team_name(
            openfootball_name
        ) == normalize_team_name(key):

            for alias in aliases:

                found = normalized_available.get(
                    normalize_team_name(alias)
                )

                if found:
                    return found

    return None


# ============================================================
# FEATURE HELPERS
# ============================================================

def get_model_feature_columns():

    columns = set()

    for collection in [
        FEATURE_COLS,
        CORNER_FEATURE_COLS,
        YELLOW_FEATURE_COLS,
    ]:

        try:
            columns.update(
                list(collection)
            )
        except Exception:
            pass

    return columns


def set_feature_if_exists(
    row: dict,
    name: str,
    value: Any,
):

    if name in get_model_feature_columns():
        row[name] = value


def calculate_current_league_context(
    data: dict,
):

    goals = []
    draws = 0
    completed = 0

    for match in get_matches(data):

        score = parse_score(
            match.get("score")
        )

        if score is None:
            continue

        home, away = score

        goals.append(
            home + away
        )

        completed += 1

        if home == away:
            draws += 1

    if not completed:
        return {
            "avg_goals": None,
            "draw_rate": None,
            "completed_matches": 0,
        }

    return {
        "avg_goals": (
            sum(goals) / len(goals)
        ),
        "draw_rate": (
            draws / completed
        ),
        "completed_matches": completed,
    }


def calculate_season_stage(
    week: int | None,
) -> float:

    if not week:
        return 0.0

    # Most of these competitions have 38 matchdays.
    return clamp(
        float(week) / 38.0,
        0.0,
        1.0,
    )


def calculate_table_features(
    standings: dict,
    home_team: str,
    away_team: str,
):

    home = standings.get(home_team)
    away = standings.get(away_team)

    if not home or not away:

        return {}

    leader_points = max(
        (
            row["points"]
            for row in standings.values()
        ),
        default=home["points"],
    )

    # Approximation of the safety line.
    sorted_table = sorted(
        standings.values(),
        key=lambda x: (
            x["points"],
            x["gd"],
            x["gf"],
        ),
        reverse=True,
    )

    safety_index = max(
        0,
        len(sorted_table) - 4,
    )

    safety_points = (
        sorted_table[safety_index]["points"]
        if sorted_table
        else 0
    )

    points_diff = (
        home["points"] -
        away["points"]
    )

    position_diff = (
        away["rank"] -
        home["rank"]
    )

    relegation_battle = (
        abs(home["rank"] - away["rank"]) <= 5
        and
        (
            home["rank"] >= len(sorted_table) - 6
            or
            away["rank"] >= len(sorted_table) - 6
        )
    )

    title_race = (
        home["rank"] <= 3
        and away["rank"] <= 3
    ) or (
        abs(
            home["points"] -
            leader_points
        ) <= 6
        and
        abs(
            away["points"] -
            leader_points
        ) <= 6
    )

    return {
        "points_diff": points_diff,
        "position_diff": position_diff,
        "is_relegation_battle": int(
            relegation_battle
        ),
        "is_title_race": int(
            title_race
        ),
        "points_to_safety": (
            min(
                home["points"],
                away["points"],
            )
            - safety_points
        ),
        "points_to_leader": (
            leader_points -
            max(
                home["points"],
                away["points"],
            )
        ),
    }


def find_last_match_date(
    data: dict,
    team: str,
):

    dates = []

    target = normalize_team_name(
        team
    )

    for match in get_matches(data):

        score = parse_score(
            match.get("score")
        )

        if score is None:
            continue

        home = normalize_team_name(
            match.get("team1", "")
        )

        away = normalize_team_name(
            match.get("team2", "")
        )

        if target not in (
            home,
            away,
        ):
            continue

        value = match_date_value(match)

        try:
            dates.append(
                datetime.strptime(
                    value,
                    "%Y-%m-%d",
                ).date()
            )
        except Exception:
            pass

    if not dates:
        return None

    return max(dates)


def calculate_rest_days(
    data: dict,
    team: str,
    match_date: str,
):

    last_date = find_last_match_date(
        data,
        team,
    )

    try:
        current_date = datetime.strptime(
            match_date[:10],
            "%Y-%m-%d",
        ).date()
    except Exception:
        return None

    if last_date is None:
        return None

    return max(
        0,
        (current_date - last_date).days,
    )


# ============================================================
# TIKAML FEATURE ROW
# ============================================================

def build_gagne_feature_row(
    home_tika: str,
    away_tika: str,
    league_code: str,
    season: str,
    match_date: str,
    week: int | None,
    openfootball_data: dict,
):

    if club_predictor is None:
        raise RuntimeError(
            "TikaML MatchPredictor is not loaded."
        )

    league_info = OPENFOOTBALL_LEAGUES[
        league_code
    ]

    league_name = league_info["name"]

    # TikaML internally creates its historical
    # rolling feature vector.
    row = club_predictor.build_feature_row(
        home_team=home_tika,
        away_team=away_tika,
        league=league_code,
        season=normalize_season(season),
        match_date=match_date,
        week=week,
        odds=None,
    )

    if isinstance(row, pd.Series):
        row = row.to_dict()
    else:
        row = dict(row)

    standings = build_standings(
        openfootball_data
    )

    table_features = calculate_table_features(
        standings,
        openfootball_data["_gagne_home_openfootball"],
        openfootball_data["_gagne_away_openfootball"],
    )

    for name, value in table_features.items():
        set_feature_if_exists(
            row,
            name,
            value,
        )

    h2h = build_h2h(
        openfootball_data,
        openfootball_data["_gagne_home_openfootball"],
        openfootball_data["_gagne_away_openfootball"],
    )

    h2h_values = {
        "h2h_win_pct_home": (
            h2h["home_win_pct"]
            if h2h["home_win_pct"] is not None
            else 0.5
        ),
        "h2h_goal_diff_home": (
            h2h["goal_diff_home"]
        ),
        "h2h_matches": (
            h2h["matches"]
        ),
    }

    for name, value in h2h_values.items():

        set_feature_if_exists(
            row,
            name,
            value,
        )

    league_context = (
        calculate_current_league_context(
            openfootball_data
        )
    )

    if league_context["avg_goals"] is not None:

        set_feature_if_exists(
            row,
            "league_avg_goals",
            league_context["avg_goals"],
        )

    if league_context["draw_rate"] is not None:

        set_feature_if_exists(
            row,
            "league_draw_rate",
            league_context["draw_rate"],
        )

    set_feature_if_exists(
        row,
        "season_stage",
        calculate_season_stage(week),
    )

    # --------------------------------------------------------
    # Current rest days
    # --------------------------------------------------------

    home_rest = calculate_rest_days(
        openfootball_data,
        openfootball_data["_gagne_home_openfootball"],
        match_date,
    )

    away_rest = calculate_rest_days(
        openfootball_data,
        openfootball_data["_gagne_away_openfootball"],
        match_date,
    )

    if home_rest is not None:
        set_feature_if_exists(
            row,
            "days_rest_home",
            home_rest,
        )

    if away_rest is not None:
        set_feature_if_exists(
            row,
            "days_rest_away",
            away_rest,
        )

    if (
        home_rest is not None
        and away_rest is not None
    ):
        set_feature_if_exists(
            row,
            "days_rest_diff",
            home_rest - away_rest,
        )

    # Remove internal helper keys.
    openfootball_data.pop(
        "_gagne_home_openfootball",
        None,
    )

    openfootball_data.pop(
        "_gagne_away_openfootball",
        None,
    )

    return row, standings, h2h


# ============================================================
# PREDICTION ENGINE
# ============================================================

def _build_feature_df(
    feature_vector: dict,
    feature_columns,
):

    row = {}

    for col in feature_columns:
        row[col] = feature_vector.get(
            col,
            np.nan,
        )

    return pd.DataFrame(
        [row],
        columns=list(feature_columns),
    )


def score_matrix(
    lambda_home: float,
    lambda_away: float,
):

    matrix = np.zeros(
        (
            MAX_GOALS + 1,
            MAX_GOALS + 1,
        )
    )

    for home_goals in range(
        MAX_GOALS + 1
    ):

        for away_goals in range(
            MAX_GOALS + 1
        ):

            matrix[
                home_goals,
                away_goals,
            ] = (
                poisson.pmf(
                    home_goals,
                    lambda_home,
                )
                *
                poisson.pmf(
                    away_goals,
                    lambda_away,
                )
            )

    total = matrix.sum()

    if total > 0:
        matrix /= total

    return matrix


def prediction_from_lambdas(
    lambda_home: float,
    lambda_away: float,
):

    matrix = score_matrix(
        lambda_home,
        lambda_away,
    )

    home_probability = float(
        np.tril(
            matrix,
            -1,
        ).sum()
    )

    away_probability = float(
        np.triu(
            matrix,
            1,
        ).sum()
    )

    draw_probability = float(
        np.trace(matrix)
    )

    # Numerical correction.
    total = (
        home_probability
        + draw_probability
        + away_probability
    )

    if total > 0:

        home_probability /= total
        draw_probability /= total
        away_probability /= total

    # Most probable exact score.
    score_index = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    recommended_home = int(
        score_index[0]
    )

    recommended_away = int(
        score_index[1]
    )

    over_1_5 = 0.0
    over_2_5 = 0.0
    over_3_5 = 0.0

    under_1_5 = 0.0
    under_2_5 = 0.0
    under_3_5 = 0.0

    for h in range(
        MAX_GOALS + 1
    ):

        for a in range(
            MAX_GOALS + 1
        ):

            p = matrix[h, a]

            total_goals = h + a

            if total_goals > 1.5:
                over_1_5 += p
            else:
                under_1_5 += p

            if total_goals > 2.5:
                over_2_5 += p
            else:
                under_2_5 += p

            if total_goals > 3.5:
                over_3_5 += p
            else:
                under_3_5 += p

    return {
        "probabilities": {
            "home": float(home_probability),
            "draw": float(draw_probability),
            "away": float(away_probability),
        },
        "expected_goals": {
            "home": float(lambda_home),
            "away": float(lambda_away),
            "total": float(
                lambda_home + lambda_away
            ),
        },
        "recommended_score": {
            "home": recommended_home,
            "away": recommended_away,
            "probability": float(
                matrix[
                    recommended_home,
                    recommended_away,
                ]
            ),
        },
        "over_under": {
            "over_1_5": float(over_1_5),
            "under_1_5": float(under_1_5),
            "over_2_5": float(over_2_5),
            "under_2_5": float(under_2_5),
            "over_3_5": float(over_3_5),
            "under_3_5": float(under_3_5),
        },
    }


def predict_with_models(
    feature_vector: dict,
):

    if models.goals is None:
        raise RuntimeError(
            "Goal model is not loaded."
        )

    feature_df = _build_feature_df(
        feature_vector,
        FEATURE_COLS,
    )

    prediction = models.goals.predict(
        feature_df
    )

    lambda_home = float(
        prediction[0]
        if isinstance(
            prediction,
            (list, tuple, np.ndarray),
        )
        else prediction
    )

    # Some versions return two values,
    # others expose a dict.
    if isinstance(prediction, dict):

        lambda_home = float(
            prediction.get(
                "home_lambda",
                prediction.get(
                    "lambda_home",
                    1.0,
                ),
            )
        )

        lambda_away = float(
            prediction.get(
                "away_lambda",
                prediction.get(
                    "lambda_away",
                    1.0,
                ),
            )
        )

    elif isinstance(
        prediction,
        (list, tuple, np.ndarray),
    ):

        values = list(
            np.asarray(prediction).flatten()
        )

        if len(values) >= 2:
            lambda_home = float(
                values[0]
            )
            lambda_away = float(
                values[1]
            )
        else:
            lambda_away = 1.0

    else:
        lambda_away = 1.0

    lambda_home = clamp(
        lambda_home,
        0.05,
        6.0,
    )

    lambda_away = clamp(
        lambda_away,
        0.05,
        6.0,
    )

    result = prediction_from_lambdas(
        lambda_home,
        lambda_away,
    )

    return result


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    probabilities: dict,
    standings: dict,
    home_team: str,
    away_team: str,
    form_home: list[str],
    form_away: list[str],
    h2h: dict,
    recommended_score: dict,
):

    home_p = float(
        probabilities["home"]
    )

    draw_p = float(
        probabilities["draw"]
    )

    away_p = float(
        probabilities["away"]
    )

    values = {
        "home": home_p,
        "draw": draw_p,
        "away": away_p,
    }

    ordered = sorted(
        values.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    predicted_side = ordered[0][0]
    top_probability = ordered[0][1]
    second_probability = ordered[1][1]

    margin = (
        top_probability -
        second_probability
    )

    home_table = standings.get(
        home_team
    )

    away_table = standings.get(
        away_team
    )

    standings_agree = False
    form_agree = False
    h2h_agree = False

    if home_table and away_table:

        if predicted_side == "home":
            standings_agree = (
                home_table["rank"]
                < away_table["rank"]
            )

        elif predicted_side == "away":
            standings_agree = (
                away_table["rank"]
                < home_table["rank"]
            )

        else:
            standings_agree = (
                abs(
                    home_table["points"]
                    -
                    away_table["points"]
                )
                <= 4
            )

    home_form_points = form_points(
        form_home
    )

    away_form_points = form_points(
        form_away
    )

    if predicted_side == "home":
        form_agree = (
            home_form_points
            > away_form_points
        )

    elif predicted_side == "away":
        form_agree = (
            away_form_points
            > home_form_points
        )

    else:
        form_agree = (
            abs(
                home_form_points
                -
                away_form_points
            )
            <= 2
        )

    if h2h.get("matches", 0) >= 2:

        h2h_home = (
            h2h["home_wins"]
            / h2h["matches"]
        )

        h2h_away = (
            h2h["away_wins"]
            / h2h["matches"]
        )

        if predicted_side == "home":
            h2h_agree = (
                h2h_home > 0.45
            )

        elif predicted_side == "away":
            h2h_agree = (
                h2h_away > 0.45
            )

        else:
            h2h_agree = (
                h2h["draws"]
                / h2h["matches"]
                >= 0.30
            )

    score_home = int(
        recommended_score["home"]
    )

    score_away = int(
        recommended_score["away"]
    )

    score_is_draw = (
        score_home == score_away
    )

    # Base model confidence.
    score = (
        35.0
        + 35.0 * top_probability
        + 45.0 * margin
    )

    # Context agreement.
    if standings_agree:
        score += 5

    if form_agree:
        score += 5

    if h2h_agree:
        score += 3

    # Exact-score contradiction penalty.
    if (
        predicted_side in ("home", "away")
        and score_is_draw
    ):
        score -= 7

    # Conservative caps.
    if top_probability < 0.40:
        score = min(
            score,
            55,
        )
    elif top_probability < 0.45:
        score = min(
            score,
            65,
        )
    elif top_probability < 0.50:
        score = min(
            score,
            75,
        )

    score = clamp(
        score,
        0,
        100,
    )

    if (
        score >= 75
        and top_probability >= 0.50
    ):
        level = "FORTE"

    elif score >= 60:
        level = "MOYENNE"

    else:
        level = "FAIBLE"

    evidence = []

    if standings_agree:
        evidence.append(
            "classement actuel favorable au choix"
        )

    if form_agree:
        evidence.append(
            "forme récente favorable au choix"
        )

    if h2h_agree:
        evidence.append(
            "H2H récent favorable au choix"
        )

    if not evidence:
        evidence.append(
            "écart statistique limité entre les équipes"
        )

    if score_is_draw and predicted_side != "draw":
        evidence.append(
            "score exact très partagé malgré le favori 1X2"
        )

    return {
        "score": round(
            float(score),
            1,
        ),
        "level": level,
        "predicted_side": predicted_side,
        "top_probability": round(
            top_probability,
            4,
        ),
        "margin": round(
            margin,
            4,
        ),
        "evidence": evidence,
    }


# ============================================================
# FORMAT GAGNE TEMPS
# ============================================================

def side_label(
    side: str,
    home_team: str,
    away_team: str,
):

    if side == "home":
        return home_team

    if side == "away":
        return away_team

    return "Match nul"


def build_gagne_prediction(
    home_team: str,
    away_team: str,
    league_code: str,
    season: str,
    match_date: str,
    week: int | None,
    data: dict,
    kickoff: str | None = None,
):

    home_tika = resolve_tika_team(
        home_team
    )

    away_tika = resolve_tika_team(
        away_team
    )

    if not home_tika:
        raise ValueError(
            f"Équipe introuvable dans TikaML: {home_team}"
        )

    if not away_tika:
        raise ValueError(
            f"Équipe introuvable dans TikaML: {away_team}"
        )

    data["_gagne_home_openfootball"] = home_team
    data["_gagne_away_openfootball"] = away_team

    feature_row, standings, h2h = (
        build_gagne_feature_row(
            home_tika=home_tika,
            away_tika=away_tika,
            league_code=league_code,
            season=season,
            match_date=match_date,
            week=week,
            openfootball_data=data,
        )
    )

    prediction = predict_with_models(
        feature_row
    )

    form_home = team_form(
        data,
        home_team,
    )

    form_away = team_form(
        data,
        away_team,
    )

    confidence = calculate_confidence(
        prediction["probabilities"],
        standings,
        home_team,
        away_team,
        form_home,
        form_away,
        h2h,
        prediction["recommended_score"],
    )

    best_pick = side_label(
        confidence["predicted_side"],
        home_team,
        away_team,
    )

    home_table = standings.get(
        home_team
    )

    away_table = standings.get(
        away_team
    )

    context = {
        "home": {
            "team": home_team,
            "tika_team": home_tika,
            "rank": (
                home_table["rank"]
                if home_table
                else None
            ),
            "points": (
                home_table["points"]
                if home_table
                else None
            ),
            "form": form_string(
                form_home
            ),
            "form_points": form_points(
                form_home
            ),
        },
        "away": {
            "team": away_team,
            "tika_team": away_tika,
            "rank": (
                away_table["rank"]
                if away_table
                else None
            ),
            "points": (
                away_table["points"]
                if away_table
                else None
            ),
            "form": form_string(
                form_away
            ),
            "form_points": form_points(
                form_away
            ),
        },
        "h2h": h2h,
    }

    return clean_json(
        {
            "match": {
                "home_team": home_team,
                "away_team": away_team,
                "league": OPENFOOTBALL_LEAGUES[
                    league_code
                ]["name"],
                "league_code": league_code,
                "country": OPENFOOTBALL_LEAGUES[
                    league_code
                ]["country"],
                "season": OPENFOOTBALL_SEASON,
                "date": match_date,
                "kickoff": kickoff,
                "week": week,
            },

            "tika": {
                "home_team": home_tika,
                "away_team": away_tika,
            },

            "prediction": {
                "best_pick": best_pick,
                "predicted_side": confidence[
                    "predicted_side"
                ],
                "probabilities": prediction[
                    "probabilities"
                ],
                "recommended_score": prediction[
                    "recommended_score"
                ],
                "expected_goals": prediction[
                    "expected_goals"
                ],
                "over_under": prediction[
                    "over_under"
                ],
            },

            "confidence": confidence,

            "context": context,

            "odds": None,
            "odds_available": False,
        }
    )


# ============================================================
# TODAY ENGINE
# ============================================================

def get_today_utc():

    return datetime.now(
        timezone.utc
    ).date().isoformat()


def extract_week(
    match: dict,
):

    value = str(
        match.get(
            "round",
            "",
        )
    )

    match_number = re.search(
        r"(\d+)",
        value,
    )

    if match_number:
        return int(
            match_number.group(1)
        )

    return None


def predict_today_matches():

    today = get_today_utc()

    predictions = []
    skipped = []
    source_errors = {}

    for league_code in OPENFOOTBALL_LEAGUES:

        try:
            data = fetch_openfootball(
                league_code
            )

        except Exception as exc:

            source_errors[
                league_code
            ] = str(exc)

            continue

        for match in get_matches(data):

            if match_date_value(match) != today:
                continue

            # Completed matches are not predictions
            # for today's upcoming list.
            if is_completed_match(match):
                continue

            home = match.get("team1")
            away = match.get("team2")

            if not home or not away:
                continue

            home_tika = resolve_tika_team(
                home
            )

            away_tika = resolve_tika_team(
                away
            )

            if not home_tika or not away_tika:

                skipped.append(
                    {
                        "league": league_code,
                        "home_team": home,
                        "away_team": away,
                        "reason": (
                            "Équipe non trouvée "
                            "dans les données TikaML"
                        ),
                        "tika_home": home_tika,
                        "tika_away": away_tika,
                    }
                )

                continue

            try:

                prediction = build_gagne_prediction(
                    home_team=home,
                    away_team=away,
                    league_code=league_code,
                    season=OPENFOOTBALL_SEASON,
                    match_date=today,
                    week=extract_week(match),
                    data=data,
                    kickoff=match.get("time"),
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
                        "league": league_code,
                        "home_team": home,
                        "away_team": away,
                        "reason": str(exc),
                    }
                )

    predictions.sort(
        key=lambda x: x[
            "confidence"
        ]["score"],
        reverse=True,
    )

    return {
        "date": today,
        "predictions": predictions,
        "skipped": skipped,
        "source_errors": source_errors,
    }


# ============================================================
# STANDARD TIKAML PREDICT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(
    request: PredictionRequest,
):

    if models.goals is None:
        raise HTTPException(
            status_code=503,
            detail="Goal model not loaded",
        )

    predictions = {}

    feature_df = _build_feature_df(
        request.feature_vector,
        FEATURE_COLS,
    )

    if "goals" in request.models:

        try:

            result = models.goals.predict(
                feature_df
            )

            predictions[
                "goals"
            ] = clean_json(result)

        except Exception as exc:

            log.exception(
                "Goal prediction failed"
            )

            predictions[
                "goals"
            ] = {
                "error": str(exc)
            }

    if (
        "corners" in request.models
        and models.corners is not None
    ):

        try:

            df = _build_feature_df(
                request.feature_vector,
                CORNER_FEATURE_COLS,
            )

            result = models.corners.predict(
                df
            )

            predictions[
                "corners"
            ] = clean_json(result)

        except Exception as exc:

            predictions[
                "corners"
            ] = {
                "error": str(exc)
            }

    if (
        "yellows" in request.models
        and models.yellows is not None
    ):

        try:

            df = _build_feature_df(
                request.feature_vector,
                YELLOW_FEATURE_COLS,
            )

            result = models.yellows.predict(
                df
            )

            predictions[
                "yellows"
            ] = clean_json(result)

        except Exception as exc:

            predictions[
                "yellows"
            ] = {
                "error": str(exc)
            }

    return PredictionResponse(
        predictions=predictions,
        model_metadata={
            "version": models.version,
            "prediction_type": request.prediction_type,
        },
    )


# ============================================================
# GAGNE TEMPS PREDICT
# ============================================================

@app.post(
    "/gagne-temps/predict",
    dependencies=[Depends(verify_api_key)],
)
async def gagne_temps_predict(
    request: GagneTempsRequest,
):

    league_code = normalize_league_code(
        request.league
    )

    if league_code not in OPENFOOTBALL_LEAGUES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Ligue non supportée. "
                "Utilise EPL, LL, SEA, BUN ou LI1."
            ),
        )

    try:

        data = fetch_openfootball(
            league_code
        )

        result = build_gagne_prediction(
            home_team=request.home_team,
            away_team=request.away_team,
            league_code=league_code,
            season=request.season,
            match_date=request.match_date,
            week=request.week,
            data=data,
        )

        return {
            "status": "success",
            "source": (
                "OpenFootball + TikaML"
            ),
            "model": (
                "TikaML MatchPredictor"
            ),
            "version": models.version,
            **result,
        }

    except Exception as exc:

        log.exception(
            "GAGNE TEMPS prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# GAGNE TEMPS TODAY
# ============================================================

@app.get(
    "/gagne-temps/today",
    dependencies=[Depends(verify_api_key)],
)
async def gagne_temps_today():

    result = predict_today_matches()

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
            "date": result["date"],
            "count": len(
                result["predictions"]
            ),
            "predictions": result[
                "predictions"
            ],
            "skipped": result[
                "skipped"
            ],
            "source_errors": result[
                "source_errors"
            ],
        }
    )


# ============================================================
# GAGNE TEMPS TOP
# ============================================================

@app.get(
    "/gagne-temps/top",
    dependencies=[Depends(verify_api_key)],
)
async def gagne_temps_top():

    result = predict_today_matches()

    top_predictions = result[
        "predictions"
    ][:5]

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
            "date": result["date"],
            "count": len(
                top_predictions
            ),
            "top_5": top_predictions,
            "skipped": result[
                "skipped"
            ],
            "source_errors": result[
                "source_errors"
            ],
        }
    )


# ============================================================
# MODEL STATUS
# ============================================================

@app.get(
    "/model-status",
    dependencies=[Depends(verify_api_key)],
)
async def model_status():

    national_loaded = False

    try:
        national_loaded = (
            getattr(
                national_api,
                "model",
                None,
            )
            is not None
        )
    except Exception:
        national_loaded = False

    return clean_json(
        {
            "status": "ok",

            "models": {
                "goals": (
                    models.goals is not None
                ),
                "corners": (
                    models.corners is not None
                ),
                "yellows": (
                    models.yellows is not None
                ),
                "version": models.version,
            },

            "gagne_temps": {
                "predictor_loaded": (
                    club_predictor
                    is not None
                ),
                "historical_matches": (
                    len(
                        club_predictor.df
                    )
                    if club_predictor is not None
                    and getattr(
                        club_predictor,
                        "df",
                        None,
                    ) is not None
                    else 0
                ),
            },

            "football_data": {
                "provider": "OpenFootball",
                "api_key_required": False,
                "source": "GitHub raw JSON",
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
                        "country": info["country"],
                        "file": info["file"],
                    }
                    for code, info
                    in OPENFOOTBALL_LEAGUES.items()
                ],
                "cache_ttl_seconds": (
                    OPENFOOTBALL_CACHE_TTL
                ),
                "source_errors": (
                    _openfootball_errors
                ),
            },

            "national": {
                "loaded": national_loaded,
            },
        }
    )


# ============================================================
# BACKFILL
# ============================================================

@app.post(
    "/backfill",
    dependencies=[Depends(verify_api_key)],
)
async def backfill(
    request: PredictionRequest,
):

    active_models = backfill_models

    if active_models.goals is None:
        active_models = models

    if active_models.goals is None:

        raise HTTPException(
            status_code=503,
            detail="Backfill goal model not loaded",
        )

    predictions = {}

    feature_df = _build_feature_df(
        request.feature_vector,
        FEATURE_COLS,
    )

    try:

        predictions[
            "goals"
        ] = clean_json(
            active_models.goals.predict(
                feature_df
            )
        )

    except Exception as exc:

        predictions[
            "goals"
        ] = {
            "error": str(exc)
        }

    if (
        "corners" in request.models
        and active_models.corners is not None
    ):

        try:

            df = _build_feature_df(
                request.feature_vector,
                CORNER_FEATURE_COLS,
            )

            predictions[
                "corners"
            ] = clean_json(
                active_models.corners.predict(
                    df
                )
            )

        except Exception as exc:

            predictions[
                "corners"
            ] = {
                "error": str(exc)
            }

    if (
        "yellows" in request.models
        and active_models.yellows is not None
    ):

        try:

            df = _build_feature_df(
                request.feature_vector,
                YELLOW_FEATURE_COLS,
            )

            predictions[
                "yellows"
            ] = clean_json(
                active_models.yellows.predict(
                    df
                )
            )

        except Exception as exc:

            predictions[
                "yellows"
            ] = {
                "error": str(exc)
            }

    return {
        "predictions": predictions,
        "model_metadata": {
            "version": models.version,
            "backfill": True,
        },
    }


# ============================================================
# HEALTH
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
    }
