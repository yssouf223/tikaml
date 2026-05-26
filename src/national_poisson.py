"""National-team goal model: time-weighted, neutral-aware Poisson strength ratings.

Selected by leave-one-World-Cup-out backtest (RPS ~0.20, beats Elo / Dixon-Coles
1x2 / LightGBM hybrid / bookmaker market). Each team has attack (alpha) and
defense (beta) parameters estimated by weighted maximum likelihood:

    lambda_home = exp(att_home + def_away + home_adv * (1 - neutral))
    lambda_away = exp(att_away + def_home)

Weights combine an exponential time-decay (half-life, recent matches matter more)
and a match-importance factor (friendlies down-weighted). A Dixon-Coles low-score
correction (rho) refines the score matrix. Output (7x7 matrix -> 1x2 + over/under
+ recommended score) is identical to the five-league goal model in inference.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

DEFAULT_HALF_LIFE = 365      # days; tuned via half-life sweep
DEFAULT_FRIENDLY_W = 0.5     # match-importance weight for friendlies
LAMBDA_CLIP = (0.15, 5.0)


class NationalTeamModel:

    def __init__(self, half_life_days=DEFAULT_HALF_LIFE,
                 friendly_weight=DEFAULT_FRIENDLY_W, l2=1e-3,
                 temperature=1.0, max_goals=7, min_competitive=3):
        self.half_life_days = half_life_days
        self.friendly_weight = friendly_weight
        self.l2 = l2
        self.temperature = temperature
        self.max_goals = max_goals
        # drop non-FIFA / regional sides (Basque Country, Catalonia, ...) that
        # only ever play friendlies: keep teams with >= this many competitive games
        self.min_competitive = min_competitive
        self.attack = {}       # team -> attack rating
        self.defense = {}      # team -> defense rating
        self.home_adv = None
        self.rho = None
        self.ref_date = None

    # ----------------------------- weights -----------------------------
    def _weights(self, dates, is_friendly, ref_date):
        age = (ref_date - dates).dt.days.values.astype(float)
        tw = np.exp(-np.log(2) * age / self.half_life_days)
        iw = np.where(is_friendly, self.friendly_weight, 1.0)
        return tw * iw

    # ----------------------------- fit -----------------------------
    def fit(self, df, ref_date=None):
        """Fit attack/defense ratings on ALL matches up to ref_date.

        Args:
            df: DataFrame with columns date, home_team, away_team,
                home_score, away_score, neutral, and a tournament column
                (used to flag friendlies).
            ref_date: reference date for time decay (default: max match date).
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["home_score"].notna() & df["away_score"].notna()]
        if ref_date is None:
            ref_date = df["date"].max()
        ref_date = pd.Timestamp(ref_date)
        df = df[df["date"] <= ref_date]
        self.ref_date = ref_date

        if "is_friendly" in df.columns:
            df["_friendly"] = df["is_friendly"].astype(bool)
        else:
            df["_friendly"] = df["tournament"].str.contains("Friendly", na=False)

        # keep only teams with enough competitive (non-friendly) matches: drops
        # non-FIFA / regional sides (Basque Country, Catalonia, ...) that only
        # ever play friendlies, without relying on team-name matching
        comp = df[~df["_friendly"]]
        cc = pd.concat([comp["home_team"], comp["away_team"]]).value_counts()
        valid = set(cc[cc >= self.min_competitive].index)
        df = df[df["home_team"].isin(valid) & df["away_team"].isin(valid)]
        is_friendly = df["_friendly"].values

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        ih = df["home_team"].map(tidx).values
        ia = df["away_team"].map(tidx).values
        hg = df["home_score"].values.astype(float)
        ag = df["away_score"].values.astype(float)
        neut = df["neutral"].astype(bool).values.astype(float)
        w = self._weights(df["date"], is_friendly, ref_date)

        att, dfn, ha = self._fit_poisson(ih, ia, hg, ag, neut, w, n)
        rho = self._fit_rho(ih, ia, hg, ag, neut, w, att, dfn, ha)

        self.attack = dict(zip(teams, att))
        self.defense = dict(zip(teams, dfn))
        self.home_adv = float(ha)
        self.rho = float(rho)
        return self

    def _fit_poisson(self, ih, ia, hg, ag, neut, w, n):
        notn = 1.0 - neut
        l2 = self.l2

        def f(p):
            att, dfn, ha = p[:n], p[n:2 * n], p[2 * n]
            lh = np.clip(np.exp(att[ih] + dfn[ia] + ha * notn), 1e-6, 30)
            ma = np.clip(np.exp(att[ia] + dfn[ih]), 1e-6, 30)
            nll = np.sum(w * (lh - hg * np.log(lh) + ma - ag * np.log(ma))) \
                + l2 * np.sum(p[:2 * n] ** 2)
            rh = w * (lh - hg); ra = w * (ma - ag)
            g_att = np.bincount(ih, rh, n) + np.bincount(ia, ra, n) + 2 * l2 * att
            g_def = np.bincount(ia, rh, n) + np.bincount(ih, ra, n) + 2 * l2 * dfn
            g_ha = np.sum(rh * notn)
            return nll, np.concatenate([g_att, g_def, [g_ha]])

        p0 = np.zeros(2 * n + 1); p0[-1] = 0.3
        res = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        att, dfn, ha = res.x[:n], res.x[n:2 * n], res.x[2 * n]
        att = att - att.mean(); dfn = dfn - dfn.mean()
        return att, dfn, ha

    def _fit_rho(self, ih, ia, hg, ag, neut, w, att, dfn, ha):
        notn = 1.0 - neut
        lh = np.clip(np.exp(att[ih] + dfn[ia] + ha * notn), 1e-6, 30)
        ma = np.clip(np.exp(att[ia] + dfn[ih]), 1e-6, 30)
        m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)

        def negll(rho):
            tau = np.ones(len(hg))
            tau[m00] = 1 - lh[m00] * ma[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + ma[m10] * rho
            tau[m11] = 1 - rho
            return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))

        return minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded").x

    # ----------------------------- predict -----------------------------
    def predict_lambdas(self, home_team, away_team, neutral=True):
        ah = self.attack.get(home_team, 0.0); dh = self.defense.get(home_team, 0.0)
        aa = self.attack.get(away_team, 0.0); da = self.defense.get(away_team, 0.0)
        lh = np.exp(ah + da + self.home_adv * (0.0 if neutral else 1.0))
        la = np.exp(aa + dh)
        return float(np.clip(lh, *LAMBDA_CLIP)), float(np.clip(la, *LAMBDA_CLIP))

    def predict_score_matrix(self, lh, la):
        n = self.max_goals
        m = np.outer(poisson.pmf(np.arange(n), lh), poisson.pmf(np.arange(n), la))
        rho = self.rho or 0.0
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
        return m / m.sum()

    def predict(self, home_team, away_team, neutral=True):
        """Full prediction dict, format identical to inference.MatchPredictor.predict."""
        lh, la = self.predict_lambdas(home_team, away_team, neutral)
        matrix = self.predict_score_matrix(lh, la)
        n = self.max_goals

        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
        probs = np.array([p_home, p_draw, p_away]); probs /= probs.sum()
        # NOTE: served 1x2 is the raw matrix-derived vector (no temperature),
        # matching the served five-league club model. Temperature was evaluated
        # (experiments/national/calibrate_temperature.py) and rejected: nested
        # leave-one-WC-out CV showed sharpening overfits and worsens out-of-sample
        # RPS/log-loss/Brier, so self.temperature is left unused at T=1.0.

        flat = matrix.flatten()
        top_scores = []
        for idx in flat.argsort()[::-1][:5]:
            i, j = divmod(idx, n)
            top_scores.append((int(i), int(j), float(flat[idx])))

        groups = _build_score_groups(matrix, n)
        oidx = int(np.argmax(probs))
        gmap = {0: "home_win", 1: "draw", 2: "away_win"}
        pg = next(g for g in groups if g["type"] == gmap[oidx])
        rec = pg["top_scores"][0]
        return {
            "score_matrix": matrix,
            "probs_1x2": probs,
            "predicted_outcome": ["home_win", "draw", "away_win"][oidx],
            "recommended_score": {"home_goals": rec[0], "away_goals": rec[1],
                                  "prob": rec[2], "label": f"{rec[0]}-{rec[1]}"},
            "lambda_home": lh,
            "lambda_away": la,
            "top_scores": top_scores,
            "score_groups": groups,
            "goals_over_under": _compute_goals_over_under(matrix, n),
            "neutral": neutral,
        }

    def team_ratings(self, top_n=None):
        rows = [{"team": t, "attack": self.attack[t], "defense": self.defense[t],
                 "strength": self.attack[t] - self.defense[t]} for t in self.attack]
        fi = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
        return fi.head(top_n) if top_n else fi

    # ----------------------------- io -----------------------------
    def save(self, directory="models/national"):
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        meta = {
            "half_life_days": self.half_life_days,
            "friendly_weight": self.friendly_weight,
            "l2": self.l2, "temperature": self.temperature,
            "max_goals": self.max_goals,
            "home_adv": self.home_adv, "rho": self.rho,
            "ref_date": str(self.ref_date.date()) if self.ref_date is not None else None,
            "attack": self.attack, "defense": self.defense,
        }
        with open(path / "ratings.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  国家队模型已保存到 {path}/  ({len(self.attack)} 队)")

    @classmethod
    def load(cls, directory="models/national"):
        with open(Path(directory) / "ratings.json") as f:
            meta = json.load(f)
        m = cls(half_life_days=meta["half_life_days"], friendly_weight=meta["friendly_weight"],
                l2=meta["l2"], temperature=meta["temperature"], max_goals=meta["max_goals"])
        m.home_adv = meta["home_adv"]; m.rho = meta["rho"]
        m.ref_date = pd.Timestamp(meta["ref_date"]) if meta["ref_date"] else None
        m.attack = meta["attack"]; m.defense = meta["defense"]
        print(f"  国家队模型已加载: {len(m.attack)} 队, home_adv={m.home_adv:.3f}, rho={m.rho:.3f}")
        return m


# --- output helpers (identical to src/inference.py for output consistency) ---
def _compute_goals_over_under(matrix, max_goals=7):
    ou = {}
    for line in [1.5, 2.5, 3.5]:
        p_over = sum(matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i + j > line)
        ou[line] = {"over": float(p_over), "under": float(1 - p_over)}
    return ou


def _build_score_groups(matrix, max_goals=7):
    groups = []
    home = sorted([(i, j, matrix[i, j]) for i in range(max_goals) for j in range(i)], key=lambda x: -x[2])
    groups.append({"type": "home_win", "label": "主胜", "total_prob": sum(s[2] for s in home),
                   "top_scores": [(s[0], s[1], s[2]) for s in home[:3]]})
    draw = sorted([(i, i, matrix[i, i]) for i in range(max_goals)], key=lambda x: -x[2])
    groups.append({"type": "draw", "label": "平局", "total_prob": sum(s[2] for s in draw),
                   "top_scores": [(s[0], s[1], s[2]) for s in draw[:3]]})
    away = sorted([(i, j, matrix[i, j]) for i in range(max_goals) for j in range(i + 1, max_goals)], key=lambda x: -x[2])
    groups.append({"type": "away_win", "label": "客胜", "total_prob": sum(s[2] for s in away),
                   "top_scores": [(s[0], s[1], s[2]) for s in away[:3]]})
    return groups
