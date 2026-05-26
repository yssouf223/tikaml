"""Tournament-OUTCOME (champion-level) backtest for national-team models.

The existing backtest.py scores match-level RPS, which is dominated by group
games and masks how well a model predicts the *eventual champion* — exactly the
quantity at stake for World-Cup title odds (the France-vs-market debate).

This tool, for each of WC 2010/2014/2018/2022:
  1. reconstructs the 8 four-team groups from the first 48 WC matches,
  2. fits a strength model on the prior 8 years (pluggable variant),
  3. Monte-Carlo simulates the 32-team tournament (group round-robin -> standard
     knockout, with the group->bracket-slot assignment RANDOMISED each sim so the
     unknown real bracket is marginalised out and all variants are judged fairly),
  4. scores -log P_champion(actual winner).

Average champion log-loss (lower = better) is the headline metric; we also print
the actual champion's predicted probability and rank. NOTE: n=4 tournaments, so
champion log-loss is low-power/indicative — read it together with match RPS.
"""

import sys

import numpy as np
import pandas as pd
import networkx as nx

import backtest as bt

CHAMPION = {2010: "Spain", 2014: "Germany", 2018: "France", 2022: "Argentina"}
RUNNER_UP = {2010: "Netherlands", 2014: "Argentina", 2018: "Croatia", 2022: "France"}

_FIFA = pd.read_csv("data/international/fifa_rankings.csv", parse_dates=["rank_date"])

# Standard 8-group knockout: Round of 16 cross-pairings (slot indices 0..7).
# (winner_g, runnerup_g) keeping the two halves apart, classic WC layout.
R16_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (1, 0), (3, 2), (5, 4), (7, 6)]
# QF feeds: which R16 matches meet; then SF, then final.
QF_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]


def reconstruct(df, yr, start):
    """Rebuild the 8 groups + group fixtures from the first 48 WC matches."""
    s = pd.Timestamp(start)
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= s) &
            (df.date < s + pd.Timedelta(days=60))].sort_values("date")
    grp = wc.head(48)
    g = nx.Graph()
    for r in grp.itertuples():
        g.add_edge(r.home_team, r.away_team)
    groups = [sorted(c) for c in nx.connected_components(g)]
    assert len(groups) == 8 and all(len(c) == 4 for c in groups), \
        f"WC{yr}: group reconstruction failed {[len(c) for c in groups]}"
    matches = [(r.home_team, r.away_team, bool(r.neutral)) for r in grp.itertuples()]
    return groups, matches


# --------------------------- model variants ---------------------------
def fit_strength(train, ref_date, cap=None):
    """Fit weighted Poisson att/def + home adv. `cap` optionally caps the goal
    margin in training (anti weak-opposition blowout-padding)."""
    teams = sorted(set(train.home_team) | set(train.away_team))
    tidx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    ih = train.home_team.map(tidx).values
    ia = train.away_team.map(tidx).values
    hg = train.home_score.values.astype(float).copy()
    ag = train.away_score.values.astype(float).copy()
    if cap is not None:
        margin = hg - ag
        capped = np.clip(margin, -cap, cap)
        lo = np.minimum(hg, ag)
        hg = np.where(margin >= 0, lo + capped, lo)        # winner = loser + capped margin
        ag = np.where(margin >= 0, lo, lo - capped)
    neut = train.neutral.values.astype(float)
    w = bt.weights(train, ref_date)
    att, dfn, ha = bt.fit_poisson(ih, ia, hg, ag, neut, w, n)
    return tidx, att, dfn, ha


def fit_fifa_blend(train, ref_date, w=0.4, cap=None):
    """Blend the model's att/def strength with a FIFA-ranking talent prior.

    FIFA points are z-scored within the pre-WC snapshot (raw scale is not era-
    comparable), rescaled to the model strength's own spread, split symmetrically
    into an attack/defence shift, and mixed in at weight `w`. Tests whether an
    external talent signal the goals model misses improves prediction. Teams with
    no FIFA match keep z=0 (fall back to the pure model)."""
    tidx, att, dfn, ha = fit_strength(train, ref_date, cap=cap)
    fifa = _FIFA[_FIFA.rank_date < ref_date]
    snap = fifa.sort_values("rank_date").groupby("country_full").tail(1)
    pts = dict(zip(snap.country_full, snap.total_points))
    teams = list(tidx)
    p = np.array([pts.get(t, np.nan) for t in teams], float)
    mu, sd_p = np.nanmean(p), np.nanstd(p)
    z = np.where(np.isnan(p), 0.0, (p - mu) / sd_p)        # missing -> neutral
    sd_strength = np.std(att - dfn)
    s_fifa = z * sd_strength                                 # FIFA strength on model scale
    att_b = (1 - w) * att + w * (s_fifa / 2)
    dfn_b = (1 - w) * dfn + w * (-s_fifa / 2)
    cov = np.mean(~np.isnan(p))
    return tidx, att_b, dfn_b, ha, cov


def lambdas_from(tidx, att, dfn, ha, home, away, neutral):
    ih = tidx.get(home); ia = tidx.get(away)
    ah, dh = (att[ih], dfn[ih]) if ih is not None else (0., 0.)
    aa, da = (att[ia], dfn[ia]) if ia is not None else (0., 0.)
    lh = np.clip(np.exp(ah + da + (0 if neutral else ha)), .15, 5)
    la = np.clip(np.exp(aa + dh), .15, 5)
    return lh, la


# --------------------------- simulation ---------------------------
def simulate_champion(groups, matches, lam_fn, n_sims=20000, seed=0):
    """Return {team: P_champion}. lam_fn(home, away, neutral) -> (lh, la)."""
    rng = np.random.default_rng(seed)
    teams = [t for g in groups for t in g]
    # precompute pairwise lambdas + neutral-venue 1X2 for knockout
    lam = {}
    p1 = {}
    for i, t in enumerate(teams):
        for u in teams:
            if t == u:
                continue
            lh, la = lam_fn(t, u, True)
            lam[(t, u)] = (lh, la)
            m = bt.score_matrix(lh, la, 0.0)
            p1[(t, u)] = (float(np.tril(m, -1).sum()), float(np.trace(m)))
    # group fixtures keep their real neutral flag
    fixtures = [(h, a, neut, *lam_fn(h, a, neut)) for (h, a, neut) in matches]
    champ = {t: 0 for t in teams}

    def ko_winner(t1, t2):
        ph, pdraw = p1[(t1, t2)]
        adv = ph + 0.5 * pdraw
        pa = 1 - ph - pdraw
        return t1 if rng.random() < adv / (adv + pa + 0.5 * pdraw) else t2

    for _ in range(n_sims):
        pts = {t: 0 for t in teams}; gd = {t: 0 for t in teams}; gf = {t: 0 for t in teams}
        for (h, a, neut, lh, la) in fixtures:
            hgs = rng.poisson(lh); ags = rng.poisson(la)
            gf[h] += hgs; gf[a] += ags; gd[h] += hgs - ags; gd[a] += ags - hgs
            if hgs > ags: pts[h] += 3
            elif hgs < ags: pts[a] += 3
            else: pts[h] += 1; pts[a] += 1
        # rank each group, take top 2
        order = list(range(8)); rng.shuffle(order)   # randomise group->bracket slot
        slot = {}
        for gi, grp in enumerate(groups):
            ranked = sorted(grp, key=lambda t: (pts[t], gd[t], gf[t], rng.random()), reverse=True)
            slot[order[gi]] = (ranked[0], ranked[1])   # (winner, runner-up)
        # Round of 16
        r16 = [ko_winner(slot[wg][0], slot[rg][1]) for (wg, rg) in R16_PAIRS]
        qf = [ko_winner(r16[a], r16[b]) for (a, b) in QF_PAIRS]
        sf = [ko_winner(qf[a], qf[b]) for (a, b) in SF_PAIRS]
        champ[ko_winner(sf[0], sf[1])] += 1
    return {t: c / n_sims for t, c in champ.items()}


def _build(name, opts, train, ref_date):
    """Return (lam_fn, info) for a variant. info may carry FIFA coverage."""
    if opts.get("fifa"):
        tidx, att, dfn, ha, cov = fit_fifa_blend(
            train, ref_date, w=opts["fifa"], cap=opts.get("cap"))
        info = {"fifa_cov": cov}
    else:
        tidx, att, dfn, ha = fit_strength(train, ref_date, cap=opts.get("cap"))
        info = {}
    return (lambda h, a, neut, _t=tidx, _at=att, _df=dfn, _ha=ha:
            lambdas_from(_t, _at, _df, _ha, h, a, neut)), info


def evaluate(variants, n_sims=20000):
    """Score every variant on champion log-loss AND match-level RPS (4 WCs)."""
    df = bt.load()
    ll = {name: [] for name in variants}       # champion log-loss per WC
    rps = {name: [] for name in variants}       # match-level RPS per WC
    detail = {name: {} for name in variants}
    cover = {}
    for yr, start in bt.WCS.items():
        s = pd.Timestamp(start)
        train = df[(df.date < s) & (df.date >= s - pd.Timedelta(days=365 * bt.TRAIN_YEARS))]
        test = df[(df.tournament == "FIFA World Cup") & (df.date >= s) &
                  (df.date < s + pd.Timedelta(days=60))]
        groups, matches = reconstruct(df, yr, start)
        for name, opts in variants.items():
            lam_fn, info = _build(name, opts, train, s)
            cover[name] = info.get("fifa_cov")
            # champion log-loss
            pch = simulate_champion(groups, matches, lam_fn, n_sims=n_sims, seed=yr)
            actual = CHAMPION[yr]
            p = pch.get(actual, 1e-6)
            rank = 1 + sum(1 for t, v in pch.items() if v > p)
            ll[name].append(-np.log(max(p, 1e-6)))
            detail[name][yr] = (p, rank)
            # match-level RPS over all WC matches
            rr = []
            for r in test.itertuples():
                lh, la = lam_fn(r.home_team, r.away_team, bool(r.neutral))
                p1x2 = bt.matrix_to_1x2(bt.score_matrix(lh, la, 0.0))
                rr.append(bt.rps_1x2(p1x2, bt.outcome_idx(r.home_score, r.away_score)))
            rps[name].append(np.mean(rr))
    return ll, rps, detail, cover


def main():
    bt.HALF_LIFE_DAYS = 365      # match production
    variants = {
        "基线(纯比分)": {},
        "大比分封顶 cap=3": {"cap": 3},
        "FIFA先验 w=0.3": {"fifa": 0.3},
        "FIFA先验 w=0.5": {"fifa": 0.5},
        "FIFA w=0.3+cap3": {"fifa": 0.3, "cap": 3},
    }
    ll, rps, detail, cover = evaluate(variants)
    print(f"\n{'='*82}")
    print("夺冠层回测: 真实冠军的预测概率/排名 + 冠军log-loss + 比赛级RPS (越低越好)")
    print(f"{'='*82}")
    hdr = f"{'变体':<18}" + "".join(f"WC{y}        " for y in bt.WCS) + "log-loss  RPS"
    print(hdr)
    print("-" * 82)
    for name in variants:
        cells = "".join(f"{detail[name][y][0]*100:4.1f}%/{detail[name][y][1]:<2d}   " for y in bt.WCS)
        print(f"{name:<18}{cells}{np.mean(ll[name]):<10.4f}{np.mean(rps[name]):.4f}")
    print("\n真实冠军: 2010西班牙 2014德国 2018法国 2022阿根廷  (P%/名 = 该届真实冠军的预测夺冠概率/在我们榜上的名次)")
    fcov = cover.get("FIFA先验 w=0.3")
    if fcov is not None:
        print(f"FIFA排名队名匹配覆盖率≈{fcov:.0%} (未匹配队按中性回退纯模型)")
    print("注: 冠军log-loss n=4届低功效; 比赛级RPS n≈256场高功效, 以RPS为主判据")


if __name__ == "__main__":
    main()
