"""Answer two questions with data:
  Q2: pooled odds-blend benefit (2010+2014, linear + logit blends, single tuned weight)
  Q3: do orthogonal features help? test confederation membership in the hybrid.
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

import backtest as bt
import backtest_odds as bo
import backtest_hybrid as bh

warnings.filterwarnings("ignore")
bt.HALF_LIFE_DAYS = 365


# ----------------------- Q2: pooled odds blend -----------------------
def pooled_odds():
    df = bt.load(); odds = bo.load_odds()
    recs = []
    for yr in [2010, 2014]:
        s = bt.WCS[yr]
        tidx, att, dfn, ha = bo.fit_for_wc(df, s)
        test = df[(df.tournament == "FIFA World Cup") & (df.date >= pd.Timestamp(s)) &
                  (df.date < pd.Timestamp(s) + pd.Timedelta(days=60))]
        for r in test.itertuples():
            o = bt.outcome_idx(r.home_score, r.away_score)
            key = pd.Timestamp(r.date).strftime("%Y-%m-%d") + "|" + \
                "|".join(sorted([str(r.home_team), str(r.away_team)]))
            if key not in odds.index:
                continue
            row = odds.loc[key]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            mkt = bo.implied_probs(row.odds_h, row.odds_d, row.odds_a) if str(row.home_team) == str(r.home_team) \
                else bo.implied_probs(row.odds_a, row.odds_d, row.odds_h)
            recs.append((bo.model_probs(r, tidx, att, dfn, ha), mkt, o))

    def lin(a): return np.mean([bt.rps_1x2(a * m + (1 - a) * k, o) for m, k, o in recs])
    def logit(a):
        out = []
        for m, k, o in recs:
            lp = a * np.log(np.clip(m, 1e-6, 1)) + (1 - a) * np.log(np.clip(k, 1e-6, 1))
            p = np.exp(lp); p /= p.sum()
            out.append(bt.rps_1x2(p, o))
        return np.mean(out)
    grid = np.linspace(0, 1, 21)
    bl_a = min(grid, key=lin); bg_a = min(grid, key=logit)
    print(f"\n{'='*60}\nQ2: 赔率融合 (2010+2014合并池, n={len(recs)})\n{'='*60}")
    print(f"  仅模型           RPS = {lin(1.0):.4f}")
    print(f"  仅市场           RPS = {lin(0.0):.4f}")
    print(f"  最佳线性融合     RPS = {lin(bl_a):.4f}  (模型权重={bl_a:.2f})")
    print(f"  最佳logit融合    RPS = {logit(bg_a):.4f}  (模型权重={bg_a:.2f})")
    print(f"  → 融合相对纯模型增益: {lin(1.0)-min(lin(bl_a),logit(bg_a)):+.4f}")


# ----------------------- Q3: confederation feature -----------------------
def conf_ablation():
    df = bt.load()
    fifa = pd.read_csv("data/international/fifa_rankings.csv", parse_dates=["rank_date"])
    latest = fifa.sort_values("rank_date").groupby("country_full").tail(1)
    conf_map = dict(zip(latest.country_full, latest.confederation))
    confs = {c: i for i, c in enumerate(sorted(set(conf_map.values())) + ["OTHER"])}

    def cof(t): return confs.get(conf_map.get(t, "OTHER"), confs["OTHER"])

    snaps = bh.build_snapshots(df, range(2004, 2023))
    feat = bh.build_features(df, snaps, 2004)
    feat["conf_h"] = feat.home_team.map(cof); feat["conf_a"] = feat.away_team.map(cof)
    cov = feat.home_team.map(lambda t: t in conf_map).mean()
    print(f"\n{'='*60}\nQ3: 加入洲际特征 (confederation 覆盖率={cov:.0%})\n{'='*60}")

    sets = {"base(实力+Elo+状态)": bh.FEATS,
            "base + 洲际": bh.FEATS + ["conf_h", "conf_a"]}
    res = {k: [] for k in sets}
    for yr, start in bt.WCS.items():
        s = pd.Timestamp(start)
        tr = feat[(feat.date < s) & (feat.date >= s - pd.Timedelta(days=365 * 10))]
        te = feat[(feat.tournament == "FIFA World Cup") & (feat.date >= s) &
                  (feat.date < s + pd.Timedelta(days=60))]
        for name, cols in sets.items():
            mh = lgb.LGBMRegressor(**bh.LGB_PARAMS).fit(tr[cols], tr.home_score)
            ma = lgb.LGBMRegressor(**bh.LGB_PARAMS).fit(tr[cols], tr.away_score)
            lh = np.clip(mh.predict(te[cols]), .15, 5); la = np.clip(ma.predict(te[cols]), .15, 5)
            rr = [bt.rps_1x2(bt.matrix_to_1x2(bt.score_matrix(lh[i], la[i], 0.0)),
                             bt.outcome_idx(te.iloc[i].home_score, te.iloc[i].away_score))
                  for i in range(len(te))]
            res[name].append(np.mean(rr))
    print(f"{'特征集':<24}" + "".join(f"WC{y:<7}" for y in bt.WCS) + "平均")
    print("-" * 60)
    for name in sets:
        print(f"{name:<24}" + "".join(f"{v:<9.4f}" for v in res[name]) + f"{np.mean(res[name]):.4f}")


if __name__ == "__main__":
    pooled_odds()
    conf_ablation()
