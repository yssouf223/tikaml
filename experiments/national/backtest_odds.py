"""Quantify the value of bookmaker odds + half-life sensitivity.

Joins Beat-the-Bookie historical odds (2005-2015) to WC 2010/2014 and compares:
  model-only  vs  market-only  vs  blend(model, market).
Also sweeps the time-decay half life for the Poisson rating model.
"""

import numpy as np
import pandas as pd
import backtest as bt


def implied_probs(oh, od, oa):
    r = np.array([1 / oh, 1 / od, 1 / oa])
    return r / r.sum()


def load_odds():
    o = pd.read_csv("data/international/odds_international_2005_2015.csv", parse_dates=["date"])
    o = o.dropna(subset=["odds_h", "odds_d", "odds_a"])
    o["key"] = o.date.dt.strftime("%Y-%m-%d") + "|" + o.apply(
        lambda r: "|".join(sorted([str(r.home_team), str(r.away_team)])), axis=1)
    return o.set_index("key")


def fit_for_wc(df, start):
    s = pd.Timestamp(start)
    train = df[(df.date < s) & (df.date >= s - pd.Timedelta(days=365 * bt.TRAIN_YEARS))].copy()
    teams = sorted(set(train.home_team) | set(train.away_team))
    tidx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ih = train.home_team.map(tidx).values; ia = train.away_team.map(tidx).values
    hg = train.home_score.values.astype(float); ag = train.away_score.values.astype(float)
    neut = train.neutral.values.astype(float)
    w = bt.weights(train, s)
    att, dfn, ha = bt.fit_poisson(ih, ia, hg, ag, neut, w, n)
    return tidx, att, dfn, ha


def model_probs(r, tidx, att, dfn, ha):
    ih = tidx.get(r.home_team); ia = tidx.get(r.away_team)
    ah_, dh_ = (att[ih], dfn[ih]) if ih is not None else (0., 0.)
    aa_, da_ = (att[ia], dfn[ia]) if ia is not None else (0., 0.)
    lh = np.clip(np.exp(ah_ + da_ + (0 if r.neutral else ha)), .15, 5)
    la = np.clip(np.exp(aa_ + dh_), .15, 5)
    return bt.matrix_to_1x2(bt.score_matrix(lh, la, 0.0))


def odds_value():
    df = bt.load()
    odds = load_odds()
    print(f"\n{'='*66}\n赔率价值评估 (WC2010/2014, BTB真实收盘赔率)\n{'='*66}")
    rows = []
    for yr in [2010, 2014]:
        s = bt.WCS[yr]
        tidx, att, dfn, ha = fit_for_wc(df, s)
        test = df[(df.tournament == "FIFA World Cup") & (df.date >= pd.Timestamp(s)) &
                  (df.date < pd.Timestamp(s) + pd.Timedelta(days=60))].copy()
        m_model, m_mkt, matched = [], [], 0
        recs = []
        for r in test.itertuples():
            o = bt.outcome_idx(r.home_score, r.away_score)
            key = pd.Timestamp(r.date).strftime("%Y-%m-%d") + "|" + \
                "|".join(sorted([str(r.home_team), str(r.away_team)]))
            if key not in odds.index:
                continue
            row = odds.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            # align odds orientation to martj42 home/away
            if str(row.home_team) == str(r.home_team):
                mkt = implied_probs(row.odds_h, row.odds_d, row.odds_a)
            else:
                mkt = implied_probs(row.odds_a, row.odds_d, row.odds_h)
            mdl = model_probs(r, tidx, att, dfn, ha)
            recs.append((mdl, mkt, o))
            matched += 1
        # blend sweep
        best_a, best_r = None, 9
        for a in np.linspace(0, 1, 11):
            rr = np.mean([bt.rps_1x2(a * m + (1 - a) * k, o) for m, k, o in recs])
            if rr < best_r:
                best_r, best_a = rr, a
        rps_model = np.mean([bt.rps_1x2(m, o) for m, k, o in recs])
        rps_mkt = np.mean([bt.rps_1x2(k, o) for m, k, o in recs])
        rows.append((yr, matched, len(test), rps_model, rps_mkt, best_r, best_a))

    print(f"{'WC':<6}{'匹配/总':<10}{'模型RPS':<10}{'市场RPS':<10}{'最佳融合':<10}{'融合α(模型权重)'}")
    print("-" * 66)
    for yr, m, t, rm, rk, rb, ba in rows:
        print(f"{yr:<6}{f'{m}/{t}':<10}{rm:<10.4f}{rk:<10.4f}{rb:<10.4f}α={ba:.1f}")
    print("\n解读: 市场RPS明显低=赔率信息强; 最佳融合<两者=模型与市场互补")


def half_life_sweep():
    df = bt.load()
    print(f"\n{'='*66}\n半衰期敏感度 (时间衰减泊松, 4届WC平均RPS)\n{'='*66}")
    orig = bt.HALF_LIFE_DAYS
    print(f"{'半衰期(天)':<12}{'平均RPS':<10}")
    print("-" * 30)
    for hl in [270, 365, 500, 730, 1095, 1825]:
        bt.HALF_LIFE_DAYS = hl
        rps_all = []
        for yr, start in bt.WCS.items():
            s = pd.Timestamp(start)
            tidx, att, dfn, ha = fit_for_wc(df, start)
            test = df[(df.tournament == "FIFA World Cup") & (df.date >= s) &
                      (df.date < s + pd.Timedelta(days=60))]
            rr = [bt.rps_1x2(model_probs(r, tidx, att, dfn, ha),
                             bt.outcome_idx(r.home_score, r.away_score))
                  for r in test.itertuples()]
            rps_all.append(np.mean(rr))
        print(f"{hl:<12}{np.mean(rps_all):<10.4f}")
    bt.HALF_LIFE_DAYS = orig


if __name__ == "__main__":
    odds_value()
    half_life_sweep()
