"""Hybrid LightGBM backtest for national teams.

Features per match (computed as-of pre-match, no leakage):
  - Poisson attack/defense ratings from a yearly snapshot (fit on prior 8y, time-decayed)
  - running Elo (chronological)
  - neutral flag, days rest, rolling goals for/against (last 5)
Two LightGBM Poisson regressors -> lambda_home/away -> score matrix -> 1X2.
Compared against the parametric time-weighted Poisson baseline.
"""

import warnings
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

import backtest as bt

warnings.filterwarnings("ignore")
bt.HALF_LIFE_DAYS = 365  # best from sweep

FEATS = ["att_h", "def_h", "att_a", "def_a", "elo_h", "elo_a", "elo_diff",
         "neutral", "rest_h", "rest_a", "gf_h", "ga_h", "gf_a", "ga_a"]
LGB_PARAMS = dict(objective="poisson", n_estimators=300, learning_rate=0.03,
                  max_depth=4, num_leaves=12, min_child_samples=40,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=0.1, verbose=-1)


def build_snapshots(df, years):
    snaps = {}
    for Y in years:
        cut = pd.Timestamp(f"{Y}-01-01")
        tr = df[(df.date < cut) & (df.date >= cut - pd.Timedelta(days=365 * 8))]
        teams = sorted(set(tr.home_team) | set(tr.away_team))
        tidx = {t: i for i, t in enumerate(teams)}
        ih = tr.home_team.map(tidx).values; ia = tr.away_team.map(tidx).values
        att, dfn, ha = bt.fit_poisson(ih, ia, tr.home_score.values.astype(float),
                                      tr.away_score.values.astype(float),
                                      tr.neutral.values.astype(float), bt.weights(tr, cut), len(teams))
        snaps[Y] = (tidx, att, dfn, ha)
    return snaps


def build_features(df, snaps, min_year):
    elo = defaultdict(lambda: 1500.0)
    last_date = {}; recent_gf = defaultdict(lambda: deque(maxlen=5)); recent_ga = defaultdict(lambda: deque(maxlen=5))
    rows = []
    for r in df.itertuples():
        Y = r.date.year
        if Y < min_year:
            # still update state
            pass
        snapY = min(max(Y, min(snaps)), max(snaps))
        tidx, att, dfn, ha = snaps[snapY]
        ih = tidx.get(r.home_team); ia = tidx.get(r.away_team)
        ah_, dh_ = (att[ih], dfn[ih]) if ih is not None else (0., 0.)
        aa_, da_ = (att[ia], dfn[ia]) if ia is not None else (0., 0.)
        eh = elo[r.home_team]; ea = elo[r.away_team]
        rest_h = (r.date - last_date[r.home_team]).days if r.home_team in last_date else 180
        rest_a = (r.date - last_date[r.away_team]).days if r.away_team in last_date else 180
        gf_h = np.mean(recent_gf[r.home_team]) if recent_gf[r.home_team] else np.nan
        ga_h = np.mean(recent_ga[r.home_team]) if recent_ga[r.home_team] else np.nan
        gf_a = np.mean(recent_gf[r.away_team]) if recent_gf[r.away_team] else np.nan
        ga_a = np.mean(recent_ga[r.away_team]) if recent_ga[r.away_team] else np.nan
        rows.append(dict(date=r.date, tournament=r.tournament,
                         home_team=r.home_team, away_team=r.away_team,
                         home_score=r.home_score, away_score=r.away_score,
                         att_h=ah_, def_h=dh_, att_a=aa_, def_a=da_,
                         elo_h=eh, elo_a=ea, elo_diff=(eh + (0 if r.neutral else 70)) - ea,
                         neutral=int(r.neutral), rest_h=min(rest_h, 365), rest_a=min(rest_a, 365),
                         gf_h=gf_h, ga_h=ga_h, gf_a=gf_a, ga_a=ga_a))
        # update state
        res = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        eh_eff = eh + (70 if not r.neutral else 0)
        exp_h = 1 / (1 + 10 ** ((ea - eh_eff) / 400))
        kk = 30 * (1 + 0.5 * np.log1p(abs(r.home_score - r.away_score)))
        elo[r.home_team] = eh + kk * (res - exp_h)
        elo[r.away_team] = ea + kk * ((1 - res) - (1 - exp_h))
        last_date[r.home_team] = r.date; last_date[r.away_team] = r.date
        recent_gf[r.home_team].append(r.home_score); recent_ga[r.home_team].append(r.away_score)
        recent_gf[r.away_team].append(r.away_score); recent_ga[r.away_team].append(r.home_score)
    return pd.DataFrame(rows)


def run():
    df = bt.load()
    snaps = build_snapshots(df, range(2004, 2023))
    feat = build_features(df, snaps, 2004)

    print(f"\n{'='*60}\nHybrid LightGBM vs 参数泊松 (留一届WC, RPS)\n{'='*60}")
    print(f"{'WC':<8}{'参数泊松':<12}{'Hybrid LGBM':<14}")
    print("-" * 40)
    par_all, hyb_all = [], []
    for yr, start in bt.WCS.items():
        s = pd.Timestamp(start)
        tr = feat[(feat.date < s) & (feat.date >= s - pd.Timedelta(days=365 * 10))].copy()
        te = feat[(feat.tournament == "FIFA World Cup") & (feat.date >= s) &
                  (feat.date < s + pd.Timedelta(days=60))].copy()
        Xtr = tr[FEATS]; Xte = te[FEATS]
        mh = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, tr.home_score)
        ma = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, tr.away_score)
        lh = np.clip(mh.predict(Xte), .15, 5); la = np.clip(ma.predict(Xte), .15, 5)
        hyb = [bt.rps_1x2(bt.matrix_to_1x2(bt.score_matrix(lh[i], la[i], 0.0)),
                          bt.outcome_idx(te.iloc[i].home_score, te.iloc[i].away_score))
               for i in range(len(te))]
        # parametric baseline (att/def from snapshot of WC year)
        tidx, att, dfn, ha = snaps[min(max(yr, min(snaps)), max(snaps))]
        par = []
        for r in te.itertuples():
            ih = tidx.get(r.home_team); ia = tidx.get(r.away_team)
            ah_, dh_ = (att[ih], dfn[ih]) if ih is not None else (0., 0.)
            aa_, da_ = (att[ia], dfn[ia]) if ia is not None else (0., 0.)
            l_h = np.clip(np.exp(ah_ + da_ + (0 if r.neutral else ha)), .15, 5)
            l_a = np.clip(np.exp(aa_ + dh_), .15, 5)
            par.append(bt.rps_1x2(bt.matrix_to_1x2(bt.score_matrix(l_h, l_a, 0.0)),
                                  bt.outcome_idx(r.home_score, r.away_score)))
        par_all.append(np.mean(par)); hyb_all.append(np.mean(hyb))
        print(f"{yr:<8}{np.mean(par):<12.4f}{np.mean(hyb):<14.4f}")
    print("-" * 40)
    print(f"{'平均':<8}{np.mean(par_all):<12.4f}{np.mean(hyb_all):<14.4f}")


if __name__ == "__main__":
    run()
