"""Prediction service API.

FastAPI server that loads trained models and serves predictions
for the tikaml-data-service.

Also provides the GAGNE TEMPS end-to-end prediction endpoint,
which builds the required feature vector automatically from
data/opta/processed/features.csv.
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
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


# ─── Config ────────────────────────────────────────────────────────

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

# API Key — set via environment variable, or auto-generate
API_KEY = os.environ.get("TIKA_API_KEY", "")

if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    log.warning(f"No TIKA_API_KEY set, generated: {API_KEY}")

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    key: str = Security(api_key_header),
):
    """Validate the API key from request header."""
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return key


# ─── Models ────────────────────────────────────────────────────────

class Models:
    goals: LGBMPoissonModel | None = None
    corners: LGBMPoissonModel | None = None
    yellows: LGBMPoissonModel | None = None
    version: str = "unknown"


models = Models()
backfill_models = Models()

# GAGNE TEMPS end-to-end predictor
club_predictor: MatchPredictor | None = None


# ─── GAGNE TEMPS request schema ───────────────────────────────────

class GagneTempsRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str
    match_date: str
    week: int | None = None


# ─── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup."""

    global club_predictor

    log.info("Loading models...")
    t0 = time.time()

    # ─── Standard TikaML models ────────────────────────────────

    models.goals = LGBMPoissonModel.load(
        str(MODEL_DIR)
    )

    log.info(
        f"  Goals model loaded "
        f"({len(models.goals.feature_cols)} features)"
    )

    if CORNER_MODEL_DIR.exists():
        models.corners = LGBMPoissonModel.load(
            str(CORNER_MODEL_DIR)
        )

        log.info(
            f"  Corners model loaded "
            f"({len(models.corners.feature_cols)} features)"
        )

    if YELLOW_MODEL_DIR.exists():
        models.yellows = LGBMPoissonModel.load(
            str(YELLOW_MODEL_DIR)
        )

        log.info(
            f"  Yellows model loaded "
            f"({len(models.yellows.feature_cols)} features)"
        )

    # ─── Model version ─────────────────────────────────────────

    meta_path = MODEL_DIR / "meta.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        models.version = (
            f"lgbm-poisson-"
            f"{len(meta.get('feature_cols', []))}f"
        )

    # ─── Backfill models ───────────────────────────────────────

    if BACKFILL_DIR.exists():

        bf_goals_dir = BACKFILL_DIR / "goals"
        bf_corners_dir = BACKFILL_DIR / "corners"
        bf_yellows_dir = BACKFILL_DIR / "yellows"

        if bf_goals_dir.exists():
            backfill_models.goals = LGBMPoissonModel.load(
                str(bf_goals_dir)
            )

        if bf_corners_dir.exists():
            backfill_models.corners = LGBMPoissonModel.load(
                str(bf_corners_dir)
            )

        if bf_yellows_dir.exists():
            backfill_models.yellows = LGBMPoissonModel.load(
                str(bf_yellows_dir)
            )

        backfill_models.version = (
            "lgbm-poisson-backfill-20260131"
        )

        log.info("  Backfill models loaded")

    # ─── National model ────────────────────────────────────────

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

    # ─── GAGNE TEMPS predictor ─────────────────────────────────

    try:

        features_path = Path(
            "data/opta/processed/features.csv"
        )

        if not features_path.exists():
            raise FileNotFoundError(
                f"features.csv not found: {features_path}"
            )

        log.info(
            f"  Loading GAGNE TEMPS data: "
            f"{features_path}"
        )

        club_predictor = MatchPredictor(
            features_path=str(features_path)
        )

        club_predictor.load_model()

        log.info(
            f"  GAGNE TEMPS predictor loaded "
            f"({len(club_predictor.df)} historical matches)"
        )

    except Exception as e:

        club_predictor = None

        log.exception(
            f"  GAGNE TEMPS predictor failed to load: {e}"
        )

    log.info(
        f"  All models loaded in "
        f"{time.time() - t0:.1f}s"
    )

    yield

    log.info("Shutting down")


# ─── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="TikaML Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


# National-team routes
app.include_router(
    national_api.router,
    dependencies=[Depends(verify_api_key)],
)


# ─── Request / Response schemas ────────────────────────────────────

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


# ─── Helpers ───────────────────────────────────────────────────────

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

    ou = {}

    for line in [1.5, 2.5, 3.5]:

        p_over = sum(
            matrix[i, j]
            for i in range(MAX_GOALS)
            for j in range(MAX_GOALS)
            if i + j > line
        )

        ou[str(line)] = {
            "over": round(p_over, 4),
            "under": round(1 - p_over, 4),
        }

    return ou


def _poisson_over_under(
    lambda_total: float,
    lines: list[float],
) -> dict:

    ou = {}

    for line in lines:

        p_over = float(
            1 - poisson.cdf(
                int(line),
                lambda_total,
            )
        )

        ou[str(line)] = {
            "over": round(p_over, 4),
            "under": round(1 - p_over, 4),
        }

    return ou


def _live_over_under(
    lambda_remaining: float,
    current_total: int,
    lines: list[float],
) -> dict:

    ou = {}

    for line in lines:

        needed = line - current_total

        if needed <= 0:

            ou[str(line)] = {
                "over": 1.0,
                "under": 0.0,
            }

        else:

            p_over = float(
                1 - poisson.cdf(
                    int(needed),
                    lambda_remaining,
                )
            )

            ou[str(line)] = {
                "over": round(p_over, 4),
                "under": round(1 - p_over, 4),
            }

    return ou


# ─── Standard prediction logic ─────────────────────────────────────

def predict_prematch(
    feature_vector: dict,
    requested_models: list[str],
    m: Models | None = None,
) -> dict:

    m = m or models

    predictions = {}

    # Goals
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
            np.tril(matrix, -1).sum()
        )

        p_draw = float(
            np.trace(matrix)
        )

        p_away = float(
            np.triu(matrix, 1).sum()
        )

        total = p_home + p_draw + p_away

        p_home /= total
        p_draw /= total
        p_away /= total

        score_matrix = [
            [
                round(float(matrix[i, j]), 4)
                for j in range(MAX_GOALS)
            ]
            for i in range(MAX_GOALS)
        ]

        best_i, best_j = divmod(
            int(np.argmax(matrix)),
            MAX_GOALS,
        )

        recommended_score = {
            "home_goals": best_i,
            "away_goals": best_j,
            "prob": round(
                float(matrix[best_i, best_j]),
                4,
            ),
            "label": f"{best_i}-{best_j}",
        }

        predictions["goals"] = {

            "home_win": round(p_home, 4),

            "draw": round(p_draw, 4),

            "away_win": round(p_away, 4),

            "expected_home": round(lh, 4),

            "expected_away": round(la, 4),

            "predicted_total": round(
                lh + la,
                4,
            ),

            "over_under": _goals_over_under(
                matrix
            ),

            "score_matrix": score_matrix,

            "recommended_score":
                recommended_score,
        }

    # Corners
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

            "over_under": _poisson_over_under(
                clh + cla,
                [8.5, 9.5, 10.5, 11.5],
            ),
        }

    # Yellows
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

            "over_under": _poisson_over_under(
                ylh + yla,
                [2.5, 3.5, 4.5, 5.5],
            ),
        }

    return predictions


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
                    and 0 <= rj < MAX_GOALS
                ):
                    matrix[i, j] = rem[ri, rj]

        if matrix.sum() > 0:
            matrix /= matrix.sum()

        score_matrix = [
            [
                round(float(matrix[i, j]), 4)
                for j in range(MAX_GOALS)
            ]
            for i in range(MAX_GOALS)
        ]

        best_i, best_j = divmod(
            int(np.argmax(matrix)),
            MAX_GOALS,
        )

        recommended_score = {
            "home_goals": best_i,
            "away_goals": best_j,
            "prob": round(
                float(matrix[best_i, best_j]),
                4,
            ),
            "label": f"{best_i}-{best_j}",
        }

        predictions["goals"] = {

            "home_win": round(
                float(live["probs_1x2"][0]),
                4,
            ),

            "draw": round(
                float(live["probs_1x2"][1]),
                4,
            ),

            "away_win": round(
                float(live["probs_1x2"][2]),
                4,
            ),

            "expected_home": round(lh, 4),

            "expected_away": round(la, 4),

            "predicted_total": round(
                lh + la,
                4,
            ),

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
                k: round(float(v), 4)
                for k, v in live[
                    "next_goal"
                ].items()
            },

            "score_matrix": score_matrix,

            "recommended_score":
                recommended_score,
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

            "over_under": _live_over_under(
                (clh + cla) * r,
                ctx.home_corners
                + ctx.away_corners,
                [8.5, 9.5, 10.5, 11.5],
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

            "over_under": _live_over_under(
                (ylh + yla) * r,
                ctx.home_yellows
                + ctx.away_yellows,
                [2.5, 3.5, 4.5, 5.5],
            ),
        }

    return predictions


# ─── GAGNE TEMPS endpoint ──────────────────────────────────────────

@app.post(
    "/gagne-temps/predict",
    dependencies=[Depends(verify_api_key)],
)
async def gagne_temps_predict(
    req: GagneTempsRequest,
):
    """
    GAGNE TEMPS end-to-end prediction.

    The user supplies only:

    - home_team
    - away_team
    - league
    - season
    - match_date
    - optional week

    TikaML automatically builds the feature vector
    from features.csv and runs the saved prediction models.
    """

    if club_predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "GAGNE TEMPS predictor is not loaded. "
                "Check data/opta/processed/features.csv "
                "and the model files."
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

    # Normalize input
    home_team = req.home_team.strip()
    away_team = req.away_team.strip()
    league = req.league.strip().upper()
    season = req.season.strip()

    if not home_team or not away_team:

        raise HTTPException(
            status_code=400,
            detail=(
                "home_team and away_team "
                "are required."
            ),
        )

    if home_team.lower() == away_team.lower():

        raise HTTPException(
            status_code=400,
            detail=(
                "home_team and away_team "
                "must be different."
            ),
        )

    # Validate against historical dataset
    df = club_predictor.df

    known_home = (
        home_team in df["home_team"].values
        or home_team in df["away_team"].values
    )

    known_away = (
        away_team in df["home_team"].values
        or away_team in df["away_team"].values
    )

    if not known_home:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown home team: {home_team}. "
                "Team name must match features.csv."
            ),
        )

    if not known_away:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown away team: {away_team}. "
                "Team name must match features.csv."
            ),
        )

    # Build prediction
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
        (time.time() - t0) * 1000,
        1,
    )

    # Convert NumPy values to JSON-safe values
    def clean(obj):

        if isinstance(obj, dict):
            return {
                str(k): clean(v)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                clean(v)
                for v in obj
            ]

        if isinstance(obj, tuple):
            return [
                clean(v)
                for v in obj
            ]

        if isinstance(obj, np.ndarray):
            return clean(
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

        if pd.isna(obj):
            return None

        return obj

    result = clean(result)

    log.info(
        f"GAGNE TEMPS "
        f"{home_team} vs {away_team} "
        f"league={league} "
        f"season={season} "
        f"elapsed={elapsed}ms"
    )

    return {
        "status": "success",

        "match": {
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "season": season,
            "match_date": req.match_date,
            "week": req.week,
        },

        "prediction": result,

        "model_metadata": {
            "engine": "TikaML MatchPredictor",
            "version": models.version,
            "data_source": (
                "data/opta/processed/features.csv"
            ),
            "elapsed_ms": elapsed,
        },
    }


# ─── Standard /predict route ───────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(req: PredictionRequest):

    """Main prediction endpoint."""

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
            "version": models.version,
            "prediction_type": req.prediction_type,
            "elapsed_ms": elapsed,
        },
    )


# ─── Backfill ──────────────────────────────────────────────────────

@app.post(
    "/backfill",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def backfill(req: PredictionRequest):

    """Backfill prediction endpoint."""

    if backfill_models.goals is None:

        raise HTTPException(
            status_code=503,
            detail="Backfill models not loaded",
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

    log.info(
        f"backfill "
        f"match_id={req.match_id} "
        f"models={list(predictions.keys())} "
        f"elapsed={elapsed}ms"
    )

    return PredictionResponse(
        predictions=predictions,

        model_metadata={
            "version": backfill_models.version,
            "prediction_type": "prematch",
            "elapsed_ms": elapsed,
        },
    )


# ─── Model status ──────────────────────────────────────────────────

@app.get("/model-status")
async def model_status():

    """Model status check endpoint."""

    return {

        "status": "ok",

        "models_loaded": {
            "goals": models.goals is not None,
            "corners": models.corners is not None,
            "yellows": models.yellows is not None,
        },

        "version": models.version,

        "backfill": {
            "loaded": (
                backfill_models.goals
                is not None
            ),
            "version": backfill_models.version,
        },

        "gagne_temps": {
            "loaded": (
                club_predictor is not None
            ),
            "features_loaded": (
                club_predictor is not None
                and club_predictor.df is not None
            ),
            "historical_matches": (
                len(club_predictor.df)
                if club_predictor is not None
                and club_predictor.df is not None
                else 0
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
