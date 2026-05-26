"""Temperature-scaling calibration for the national-team goal model.

Temperature scaling (lgbm_poisson._apply_temperature) sharpens / softens the
1x2 vector. This script asks two questions on leave-one-World-Cup-out folds
(2010/14/18/22):

  1. In-sample sweep — what T minimises pooled 1x2 RPS? (OPTIMISTIC: T is tuned
     on the very matches it is scored on, so the headline "gain" is an upper
     bound, not an honest out-of-sample estimate.)
  2. Nested LOO-on-T — the HONEST verdict: for each held-out WC pick T on the
     OTHER three WCs, then score the held-out WC. Pool the four held-out folds
     and compare against raw (T=1). This is the number to trust when deciding
     whether tempering genuinely helps.

NOTE: the model hyperparameters below mirror the production training config in
models/national/ratings.json (half-life 365, friendly weight 0.5). Keep them in
sync if the production model is ever retrained with different settings.

Temperature is a pure post-hoc transform on predict() output, so applying it
needs no retraining — `--apply` rewrites the `temperature` field in
models/national/ratings.json with the in-sample T*.

Run from anywhere:
    python3 experiments/national/calibrate_temperature.py            # report only
    python3 experiments/national/calibrate_temperature.py --apply    # + write ratings.json
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make `src` importable regardless of CWD
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.national_poisson import NationalTeamModel  # noqa: E402

DATA = ROOT / "data/international/international_results.csv"
MODEL_DIR = ROOT / "models/national"
WCS = {2010: "2010-06-11", 2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20"}
HALF_LIFE = 365         # production config (mirror ratings.json)
FRIENDLY_W = 0.5
MIN_COMPETITIVE = 3
T_GRID = np.round(np.arange(0.70, 1.301, 0.01), 3)


def load():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df.home_score.notna() & df.away_score.notna()].copy()
    df["home_score"] = df.home_score.astype(int)
    df["away_score"] = df.away_score.astype(int)
    df["neutral"] = df.neutral.astype(bool)
    return df.sort_values("date").reset_index(drop=True)


def outcome_idx(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def rps_1x2(p, o):
    """Ranked probability score for ordered [home, draw, away] (lower better)."""
    e = np.zeros(3); e[o] = 1.0
    cp = ce = c = 0.0
    for i in range(2):
        cp += p[i]; ce += e[i]
        c += (cp - ce) ** 2
    return c / 2.0


def apply_T(probs, T):
    """Temperature scaling on (N, 3) probabilities (matches lgbm_poisson)."""
    if T == 1.0:
        return probs
    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    scaled = log_p / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_p = np.exp(scaled)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def collect_per_wc():
    """Leave-one-WC-out: raw (T=1) 1x2 probs + outcomes, keyed by WC year."""
    df = load()
    folds = {}
    for yr, start in WCS.items():
        s = pd.Timestamp(start)
        train = df[df.date < s]
        # T=1.0 so predict() returns raw probs; the sweeps apply T post-hoc
        m = NationalTeamModel(half_life_days=HALF_LIFE, friendly_weight=FRIENDLY_W,
                              min_competitive=MIN_COMPETITIVE, temperature=1.0)
        m.fit(train, ref_date=s)
        test = df[(df.tournament == "FIFA World Cup") & (df.date >= s) &
                  (df.date < s + pd.Timedelta(days=60))]
        probs = np.array([m.predict(r.home_team, r.away_team, neutral=bool(r.neutral))["probs_1x2"]
                          for r in test.itertuples()])
        outcomes = np.array([outcome_idx(r.home_score, r.away_score) for r in test.itertuples()])
        folds[yr] = (probs, outcomes)
        print(f"  WC{yr}: {len(outcomes)} 场 (训练 {len(train)} 场)")
    return folds


def metrics(probs, outcomes):
    rps = np.mean([rps_1x2(p, o) for p, o in zip(probs, outcomes)])
    pc = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-8, 1.0)
    logloss = -np.mean(np.log(pc))
    onehot = np.zeros_like(probs); onehot[np.arange(len(outcomes)), outcomes] = 1.0
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))
    return rps, logloss, brier


def best_T_on(probs, outcomes):
    """T minimising RPS over the given pooled set."""
    return min(T_GRID, key=lambda T: np.mean(
        [rps_1x2(p, o) for p, o in zip(apply_T(probs, T), outcomes)]))


def run(apply=False):
    print("世界杯温度校准 (生产配置: 半衰期=365, 友谊赛权重=0.5)")
    folds = collect_per_wc()
    all_p = np.concatenate([folds[y][0] for y in WCS])
    all_o = np.concatenate([folds[y][1] for y in WCS])
    print(f"\n样本外样本: {len(all_o)} 场\n")

    # --- 1. in-sample sweep (optimistic) -------------------------------------
    sweep = [(T, np.mean([rps_1x2(p, o) for p, o in zip(apply_T(all_p, T), all_o)]))
             for T in T_GRID]
    best_T = min(sweep, key=lambda x: x[1])[0]
    print("【样本内扫描】(乐观: T 在被评分的同一批数据上选, 仅作上界参考)")
    print(f"{'T':>6}  {'RPS':>9}")
    for T, rps in sweep:
        if T in (T_GRID[0], 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, best_T, T_GRID[-1]):
            mark = "  <- T*" if T == best_T else ("   (T=1)" if T == 1.0 else "")
            print(f"{T:>6.2f}  {rps:>9.5f}{mark}")
    print(f"  样本内最优 T* = {best_T:.2f}")

    # --- 2. nested LOO-on-T (honest verdict) ---------------------------------
    # for each held-out WC, pick T on the other three, then score the held-out
    cv_probs, cv_T = [], {}
    for yr in WCS:
        tr_p = np.concatenate([folds[y][0] for y in WCS if y != yr])
        tr_o = np.concatenate([folds[y][1] for y in WCS if y != yr])
        T_yr = best_T_on(tr_p, tr_o)
        cv_T[yr] = T_yr
        cv_probs.append(apply_T(folds[yr][0], T_yr))   # held-out, T from other folds
    cv_probs = np.concatenate(cv_probs)

    raw = metrics(all_p, all_o)                 # T = 1.0 baseline
    cv = metrics(cv_probs, all_o)               # nested-CV tempered
    print("\n【嵌套留一交叉验证】(诚实: 每届的 T 只用其它三届选出 -> 真正样本外)")
    print(f"  各届选出的 T: " + ", ".join(f"WC{y}={t:.2f}" for y, t in cv_T.items()))
    print(f"{'='*54}")
    print(f"{'指标':<14}{'原始 T=1':>13}{'嵌套CV加温':>15}{'变化':>11}")
    print("-" * 54)
    for name, r, c in zip(("1x2 RPS", "log-loss", "Brier"), raw, cv):
        better = "更好" if c < r else ("更差" if c > r else "持平")
        print(f"{name:<14}{r:>13.5f}{c:>15.5f}{c-r:>+9.5f} {better}")
    print("=" * 54)
    drps = cv[0] - raw[0]
    verdict = ("加温诚实地更好" if drps < -1e-4 else
               "加温与原始基本持平 (增益在噪声内)" if abs(drps) <= 1e-4 else
               "加温诚实地更差")
    print(f"\n诚实结论: {verdict}  (样本外 RPS 变化 {drps:+.5f})")

    if apply:
        import json
        path = MODEL_DIR / "ratings.json"
        meta = json.loads(path.read_text())
        old = meta.get("temperature")
        meta["temperature"] = round(float(best_T), 3)
        path.write_text(json.dumps(meta, indent=2))
        print(f"\n已写入 {path}:  temperature {old} -> {meta['temperature']} (样本内 T*)")
    else:
        print("\n(只读模式; 加 --apply 把样本内 T* 写进 models/national/ratings.json)")
    return best_T


if __name__ == "__main__":
    run(apply="--apply" in sys.argv)
