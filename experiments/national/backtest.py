"""Leave-one-World-Cup-out backtest for national-team goal models.

Compares several methods by RPS on WC 2010/2014/2018/2022 finals.
All methods are neutral-venue aware (home advantage gated by the neutral flag)
and use time-decay + match-importance weighting where applicable.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

DATA = "data/international/international_results.csv"
WCS = {2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}
TRAIN_YEARS = 8          # training window before each WC
HALF_LIFE_DAYS = 500     # time-decay half life
FRIENDLY_W = 0.5         # match-importance weight for friendlies
MAX_GOALS = 10


# ----------------------------- data -----------------------------
def load():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df.home_score.notna()].copy()
    df["home_score"] = df.home_score.astype(int)
    df["away_score"] = df.away_score.astype(int)
    df["neutral"] = df.neutral.astype(bool)
    df["is_friendly"] = df.tournament.str.contains("Friendly", na=False)
    return df.sort_values("date").reset_index(drop=True)


def weights(df, ref_date):
    age = (ref_date - df.date).dt.days.values.astype(float)
    tw = np.exp(-np.log(2) * age / HALF_LIFE_DAYS)
    iw = np.where(df.is_friendly.values, FRIENDLY_W, 1.0)
    return tw * iw


def rps_1x2(p, outcome):
    """Ranked probability score for ordered [home, draw, away]."""
    e = np.zeros(3); e[outcome] = 1.0
    c = 0.0
    cp = ce = 0.0
    for i in range(2):
        cp += p[i]; ce += e[i]
        c += (cp - ce) ** 2
    return c / 2.0


def outcome_idx(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def matrix_to_1x2(m):
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def score_matrix(lh, la, rho=0.0):
    m = np.outer(poisson.pmf(np.arange(MAX_GOALS), lh),
                 poisson.pmf(np.arange(MAX_GOALS), la))
    if rho != 0:
        m[0, 0] *= 1 - lh * la * rho
        m[0, 1] *= 1 + lh * rho
        m[1, 0] *= 1 + la * rho
        m[1, 1] *= 1 - rho
    return m / m.sum()


# ----------------------- weighted Poisson / Dixon-Coles -----------------------
def fit_poisson(idx_h, idx_a, hg, ag, neut, w, n, l2=1e-3):
    """Weighted Poisson attack/defense ratings + home advantage (neutral-gated)."""
    notn = 1.0 - neut

    def f(p):
        att, dfn, ha = p[:n], p[n:2 * n], p[2 * n]
        lh = np.clip(np.exp(att[idx_h] + dfn[idx_a] + ha * notn), 1e-6, 30)
        ma = np.clip(np.exp(att[idx_a] + dfn[idx_h]), 1e-6, 30)
        nll = np.sum(w * (lh - hg * np.log(lh) + ma - ag * np.log(ma))) \
            + l2 * np.sum(p[:2 * n] ** 2)
        rh = w * (lh - hg); ra = w * (ma - ag)
        g_att = np.bincount(idx_h, rh, n) + np.bincount(idx_a, ra, n) + 2 * l2 * att
        g_def = np.bincount(idx_a, rh, n) + np.bincount(idx_h, ra, n) + 2 * l2 * dfn
        g_ha = np.sum(rh * notn)
        return nll, np.concatenate([g_att, g_def, [g_ha]])

    p0 = np.zeros(2 * n + 1); p0[-1] = 0.3
    res = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 400})
    att, dfn, ha = res.x[:n], res.x[n:2 * n], res.x[2 * n]
    att -= att.mean(); dfn -= dfn.mean()
    return att, dfn, ha


def fit_rho(idx_h, idx_a, hg, ag, neut, w, att, dfn, ha):
    notn = 1.0 - neut
    lh = np.clip(np.exp(att[idx_h] + dfn[idx_a] + ha * notn), 1e-6, 30)
    ma = np.clip(np.exp(att[idx_a] + dfn[idx_h]), 1e-6, 30)
    m00 = (hg == 0) & (ag == 0); m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0); m11 = (hg == 1) & (ag == 1)

    def negll(rho):
        tau = np.ones(len(hg))
        tau[m00] = 1 - lh[m00] * ma[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + ma[m10] * rho
        tau[m11] = 1 - rho
        tau = np.clip(tau, 1e-9, None)
        return -np.sum(w * np.log(tau))

    r = minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded")
    return r.x


# ----------------------------- Elo -----------------------------
def elo_ratings(train, ha=70.0, k=30.0):
    elo = {}
    for r in train.itertuples():
        eh = elo.get(r.home_team, 1500.0); ea = elo.get(r.away_team, 1500.0)
        eh_eff = eh + (ha if not r.neutral else 0.0)
        exp_h = 1 / (1 + 10 ** ((ea - eh_eff) / 400))
        res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        gd = abs(r.home_score - r.away_score)
        kk = k * (1 + 0.5 * np.log1p(gd))
        elo[r.home_team] = eh + kk * (res - exp_h)
        elo[r.away_team] = ea + kk * ((1 - res) - (1 - exp_h))
    return elo


# ----------------------------- backtest -----------------------------
def run():
    df = load()
    methods = ["Uniform", "Elo+Poisson", "Poisson(DC ρ=0)", "Dixon-Coles"]
    results = {m: {} for m in methods}

    # global goal bases (for Elo->lambda)
    rec = df[df.date >= "2002-01-01"]
    base_nn = (rec[~rec.neutral].home_score.mean(), rec[~rec.neutral].away_score.mean())
    base_nt = rec[rec.neutral].home_score.mean()  # symmetric on neutral

    for yr, start in WCS.items():
        s = pd.Timestamp(start)
        train = df[(df.date < s) & (df.date >= s - pd.Timedelta(days=365 * TRAIN_YEARS))].copy()
        test = df[(df.tournament == "FIFA World Cup") & (df.date >= s) &
                  (df.date < s + pd.Timedelta(days=60))].copy()

        teams = sorted(set(train.home_team) | set(train.away_team))
        tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        ih = train.home_team.map(tidx).values
        ia = train.away_team.map(tidx).values
        hg = train.home_score.values.astype(float)
        ag = train.away_score.values.astype(float)
        neut = train.neutral.values.astype(float)
        w = weights(train, s)

        att, dfn, ha = fit_poisson(ih, ia, hg, ag, neut, w, n)
        rho = fit_rho(ih, ia, hg, ag, neut, w, att, dfn, ha)
        elo = elo_ratings(train)
        c_elo = 0.0025  # elo-diff -> log-lambda scale

        def strength(team):
            i = tidx.get(team)
            return (att[i], dfn[i]) if i is not None else (0.0, 0.0)

        acc = {m: [] for m in methods}
        for r in test.itertuples():
            o = outcome_idx(r.home_score, r.away_score)
            nflag = r.neutral
            # Uniform
            acc["Uniform"].append(rps_1x2(np.array([1/3, 1/3, 1/3]), o))
            # Dixon-Coles / Poisson
            ah_, dh_ = strength(r.home_team); aa_, da_ = strength(r.away_team)
            lh = np.exp(ah_ + da_ + (ha if not nflag else 0.0))
            la = np.exp(aa_ + dh_)
            lh, la = np.clip([lh, la], 0.15, 5.0)
            acc["Poisson(DC ρ=0)"].append(rps_1x2(matrix_to_1x2(score_matrix(lh, la, 0.0)), o))
            acc["Dixon-Coles"].append(rps_1x2(matrix_to_1x2(score_matrix(lh, la, rho)), o))
            # Elo
            eh = elo.get(r.home_team, 1500.0); ea = elo.get(r.away_team, 1500.0)
            diff = (eh + (0 if nflag else 70.0)) - ea
            if nflag:
                lh_e = base_nt * np.exp(c_elo * diff); la_e = base_nt * np.exp(-c_elo * diff)
            else:
                lh_e = base_nn[0] * np.exp(c_elo * diff); la_e = base_nn[1] * np.exp(-c_elo * diff)
            lh_e, la_e = np.clip([lh_e, la_e], 0.15, 5.0)
            acc["Elo+Poisson"].append(rps_1x2(matrix_to_1x2(score_matrix(lh_e, la_e, 0.0)), o))

        for m in methods:
            results[m][yr] = np.mean(acc[m])

    # report
    print(f"\n{'='*70}")
    print(f"留一届世界杯回测 RPS (越低越好)  半衰期={HALF_LIFE_DAYS}天 友谊赛权重={FRIENDLY_W}")
    print(f"{'='*70}")
    hdr = f"{'方法':<20}" + "".join(f"WC{y:<8}" for y in WCS) + "平均"
    print(hdr)
    print("-" * 70)
    for m in methods:
        row = f"{m:<20}" + "".join(f"{results[m][y]:<10.4f}" for y in WCS)
        row += f"{np.mean(list(results[m].values())):.4f}"
        print(row)
    print(f"\n参考: 五大联赛进球模型 RPS=0.1968 (俱乐部, 信号更强)")


if __name__ == "__main__":
    run()
