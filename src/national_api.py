"""National-team (World Cup 2026) prediction & simulation API.

A self-contained FastAPI router mounted alongside the club models in server.py.
The national model needs no feature engineering (all inputs are self-computed
from the trained strength ratings), so it takes team names directly rather than
a feature vector. Mounted under /national with the same X-API-Key auth as
/predict. The goals block mirrors the club /predict "goals" shape (1X2 +
over/under + 7x7 score matrix + recommended score, plus live fields) so the
frontend parses both identically.
"""

import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src.national_poisson import NationalTeamModel
from src.national_simulate import TournamentSimulator, OFFICIAL_GROUPS, derive_bracket
from src.national_live import live_predict

MAX_GOALS = 7
router = APIRouter(prefix="/national", tags=["national"])


class _State:
    model: NationalTeamModel | None = None
    sim: TournamentSimulator | None = None
    groups: dict | None = None
    matches: list | None = None
    fixtures: set | None = None       # canonical (home, away) pairs for orientation check


state = _State()


def load_national(model_dir: str = "models/national") -> NationalTeamModel:
    """Load the national model + bundled WC2026 fixtures (call once at startup).

    Fixtures are bundled as JSON so the container needs no access to the
    gitignored data/international CSV."""
    model = NationalTeamModel.load(model_dir)
    sim = TournamentSimulator(model)
    with open(Path(model_dir) / "wc2026_fixtures.json") as f:
        fx = json.load(f)
    matches = [(h, a, bool(n)) for h, a, n in fx]
    # assign all-or-nothing so a mid-load failure never leaves a half-initialised state
    state.model = model
    state.sim = sim
    state.groups = {gl: list(ts) for gl, ts in OFFICIAL_GROUPS.items()}
    state.matches = matches
    state.fixtures = {(h, a) for (h, a, _) in matches}
    return model


# ─── schemas ───────────────────────────────────────────────────────

class NationalPredictRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True                 # WC games are neutral except host games
    prediction_type: str = "prematch"    # "prematch" | "live"
    minute: int = 0
    home_goals: int = 0
    away_goals: int = 0
    home_red_cards: int = 0
    away_red_cards: int = 0
    # live only: 90 = regulation, 120 = knockout tie gone to extra time. The
    # caller switches to 120 when the live feed reports extra time so the engine
    # keeps a remaining-goals rate through minute 120 instead of treating the
    # match as decided at 90.
    total_minutes: int = 90

    @field_validator("total_minutes")
    @classmethod
    def _check_total_minutes(cls, v: int) -> int:
        if v not in (90, 120):
            raise ValueError("total_minutes must be 90 (regulation) or 120 (extra time)")
        return v


class NationalSimulateRequest(BaseModel):
    n_sims: int = Field(10000, ge=1, le=100000)   # bounded to avoid runaway compute
    seed: int = 0
    # group results already played: [[home, away, home_goals, away_goals], ...]
    played: list[list] | None = None
    # knockout results already decided: {match_no: winner_team}
    played_ko: dict[int, str] | None = None
    # fixed R32 bracket {match_no: [team1, team2]}; if given, simulate from it
    bracket: dict[int, list[str]] | None = None


class NationalBracketRequest(BaseModel):
    # complete group results [[home, away, home_goals, away_goals], ...]; must
    # cover every group match (each team 3 games) or the bracket can't be derived
    played: list[list]
    seed: int = 0                        # tie-break the rare multi-solution third slotting


# ─── helpers ───────────────────────────────────────────────────────

def _ou(ou: dict) -> dict:
    return {str(k): {"over": round(float(v["over"]), 4),
                     "under": round(float(v["under"]), 4)} for k, v in ou.items()}


def _matrix_list(m) -> list:
    m = np.asarray(m)
    return [[round(float(m[i, j]), 4) for j in range(MAX_GOALS)] for i in range(MAX_GOALS)]


def _parse_played(rows: list) -> dict:
    """Validate group results and normalise to the fixture's home/away orientation.

    Accepts [home, away, home_goals, away_goals]; if the pair is given reversed
    relative to the schedule, the score is swapped to match. Raises HTTP 400 on
    malformed rows or unknown fixtures (silent mis-keying would otherwise produce
    plausible-but-wrong odds)."""
    out = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise HTTPException(status_code=400,
                                detail=f"played row must be [home, away, home_goals, away_goals]: {row}")
        h, a, hg, ag = row
        try:
            hg, ag = int(hg), int(ag)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"played scores must be integers: {row}")
        if (h, a) in state.fixtures:
            out[(h, a)] = (hg, ag)
        elif (a, h) in state.fixtures:
            out[(a, h)] = (ag, hg)                    # normalise to fixture orientation
        else:
            raise HTTPException(status_code=400, detail=f"not a WC2026 group fixture: {h} vs {a}")
    return out


def _parse_bracket(raw: dict) -> dict:
    """Validate a client-supplied R32 bracket {match_no: [team1, team2]}."""
    bracket = {}
    for k, v in raw.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise HTTPException(status_code=400,
                                detail=f"bracket match {k} must be [team1, team2]: {v}")
        bracket[int(k)] = (v[0], v[1])
    return bracket


def _prematch_block(home: str, away: str, neutral: bool) -> dict:
    r = state.model.predict(home, away, neutral)
    p = r["probs_1x2"]
    return {
        "home_win": round(float(p[0]), 4),
        "draw": round(float(p[1]), 4),
        "away_win": round(float(p[2]), 4),
        "expected_home": round(float(r["lambda_home"]), 4),
        "expected_away": round(float(r["lambda_away"]), 4),
        "predicted_total": round(float(r["lambda_home"] + r["lambda_away"]), 4),
        "over_under": _ou(r["goals_over_under"]),
        "score_matrix": _matrix_list(r["score_matrix"]),
        "recommended_score": r["recommended_score"],
    }


def _live_block(req: NationalPredictRequest) -> dict:
    out = live_predict(state.model, req.home_team, req.away_team, req.neutral,
                       req.minute, req.home_goals, req.away_goals,
                       home_red_cards=req.home_red_cards,
                       away_red_cards=req.away_red_cards,
                       total_minutes=req.total_minutes)
    p = out["probs_1x2"]
    lh, la = out["lambda_prematch"]
    lrh, lra = out["lambda_remaining"]
    # rebuild the full final-score matrix from the remaining matrix + current score
    rem = np.asarray(out["remaining_matrix"])
    matrix = np.zeros((MAX_GOALS, MAX_GOALS))
    for i in range(MAX_GOALS):
        for j in range(MAX_GOALS):
            ri, rj = i - req.home_goals, j - req.away_goals
            if 0 <= ri < rem.shape[0] and 0 <= rj < rem.shape[1]:
                matrix[i, j] = rem[ri, rj]
    if matrix.sum() > 0:
        matrix /= matrix.sum()
    else:
        # match effectively decided (minute >= 90, or >= MAX_GOALS goals): the
        # current score is the outcome — avoid a degenerate all-zeros "0-0" matrix
        ci, cj = min(req.home_goals, MAX_GOALS - 1), min(req.away_goals, MAX_GOALS - 1)
        matrix[ci, cj] = 1.0
    bi, bj = divmod(int(np.argmax(matrix)), MAX_GOALS)
    return {
        "home_win": round(float(p[0]), 4),
        "draw": round(float(p[1]), 4),
        "away_win": round(float(p[2]), 4),
        "expected_home": round(float(lh), 4),
        "expected_away": round(float(la), 4),
        "predicted_total": round(float(lh + la), 4),
        "over_under": _ou(out["over_under"]),
        "lambda_remaining_home": round(float(lrh), 4),
        "lambda_remaining_away": round(float(lra), 4),
        "next_goal": {k: round(float(v), 4) for k, v in out["next_goal"].items()},
        "score_matrix": _matrix_list(matrix),
        "recommended_score": {"home_goals": bi, "away_goals": bj,
                              "prob": round(float(matrix[bi, bj]), 4),
                              "label": f"{bi}-{bj}"},
    }


# ─── routes ────────────────────────────────────────────────────────

@router.post("/predict")
async def national_predict(req: NationalPredictRequest):
    """Single national-team match (prematch or live). Output mirrors the club
    /predict 'goals' block."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="National model not loaded")
    block = _live_block(req) if req.prediction_type == "live" \
        else _prematch_block(req.home_team, req.away_team, req.neutral)
    return {
        "predictions": {"goals": block},
        "model_metadata": {
            "version": "national-poisson",
            "prediction_type": req.prediction_type,
            "match": f"{req.home_team} vs {req.away_team}",
            "neutral": req.neutral,
        },
    }


@router.post("/simulate")
async def national_simulate(req: NationalSimulateRequest):
    """World Cup tournament simulation -> advance / round-reach / title odds.

    Pre-tournament: send nothing. In-tournament: send `played` (group results)
    and/or `played_ko` (decided knockout matches) — or a fixed `bracket` — to
    recompute the latest odds after every result."""
    if state.sim is None:
        raise HTTPException(status_code=503, detail="National model not loaded")
    played = _parse_played(req.played) if req.played else None
    played_ko = {int(k): v for k, v in req.played_ko.items()} if req.played_ko else None
    try:
        if req.bracket:
            # explicit R32 bracket (e.g. FIFA's actual draw) + any decided knockouts
            table = state.sim.simulate_from_bracket(
                _parse_bracket(req.bracket), played_ko=played_ko,
                n_sims=req.n_sims, seed=req.seed)
        elif played_ko:
            # knockout stage: fix the real bracket from the (complete) group results
            # so the knockout pins apply to stable slots, not a per-sim reshuffle
            if not played:
                raise HTTPException(status_code=400,
                                    detail="played_ko requires complete group results in `played` "
                                           "(or pass an explicit `bracket`)")
            bracket = derive_bracket(state.groups, played, seed=req.seed)
            table = state.sim.simulate_from_bracket(bracket, played_ko=played_ko,
                                                    n_sims=req.n_sims, seed=req.seed)
        else:
            # pre-tournament or group stage in progress
            table = state.sim.simulate(state.groups, state.matches, n_sims=req.n_sims,
                                       played=played, seed=req.seed)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=f"invalid simulation input: {e}")
    return {"n_sims": req.n_sims, "teams": table.round(4).to_dict(orient="records")}


@router.post("/bracket")
async def national_bracket(req: NationalBracketRequest):
    """Resolve the 16 Round-of-32 match-ups from a COMPLETE group stage.

    Applies FIFA's official template + the constraint-respecting best-third
    slotting (the same `derive_bracket` /simulate uses internally), exposed so
    the data pipeline can fill its bracket table without re-implementing the
    495-combination third-place matching. Returns {match_no: [team1, team2]}
    for matches 73-88; feed it straight back to /simulate's `bracket` for
    knockout odds consistent with the displayed tree. 400 if the group stage is
    incomplete or a fixture is unknown."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="National model not loaded")
    played = _parse_played(req.played)
    try:
        bracket = derive_bracket(state.groups, played, seed=req.seed)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=f"cannot derive bracket: {e}")
    return {"bracket": {str(m): [t1, t2] for m, (t1, t2) in bracket.items()}}


@router.get("/ratings")
async def national_ratings(top: int = Query(30, ge=1, le=300)):
    """Team strength table (attack / defense / strength), strongest first."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="National model not loaded")
    df = state.model.team_ratings(top)
    return {"total_teams": len(state.model.attack),
            "ratings": df.round(4).to_dict(orient="records")}
