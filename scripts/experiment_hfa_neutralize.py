"""主场优势中立化实验 —— 「五大联赛也不考虑主场优势会怎样？」

动机（2026-07-28 讨论）
----------------------
世界杯模型能 pick 平局是因为中立场（λh≈λa）。问题：联赛模型把主场优势
去掉（人为中立化）会更好还是更差？做成连续刻度顺带回答一个更细的问题：
**模型隐含的主场优势幅度是否恰好**——如果最优点在 s=0，说明现状正确；
在 s>0，说明模型高估了主场优势，缩一点反而更好。

方法
----
forward-chain（harness 协议）训练后取每场 (λh, λa)，在 log 空间做对称中立化：
    λh' = λh · h^(−s/2)，λa' = λa · h^(+s/2)
其中 h = exp(mean ln λh − mean ln λa) 是该测试季模型隐含的乘性 HFA，
s ∈ {−0.25, 0, 0.25, 0.5, 0.75, 1.0}：s=0 现状，s=1 完全中立，s<0 放大主场优势。
同一份 λ 上做变换 → 变换之间完全配对、无训练随机性；配对 ΔRPS ± SE。

用法
----
    .venv/bin/python scripts/experiment_hfa_neutralize.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, LGB_PARAMS  # noqa: E402
from src.evaluation import match_outcome  # noqa: E402

FEATURES = ROOT / "data/opta/processed/features_v3.csv"
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
S_GRID = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
RHO, TEMP = -0.108, 0.90


def probs_from_lambdas(lh_arr, la_arr, max_goals=7):
    """向量化 DC 矩阵 → 1x2 概率（复刻 LGBMPoissonModel.predict_1x2）。"""
    n = len(lh_arr)
    i = np.arange(max_goals)
    ph = poisson.pmf(i[None, :], lh_arr[:, None])       # (n, 7) home pmf
    pa = poisson.pmf(i[None, :], la_arr[:, None])
    M = ph[:, :, None] * pa[:, None, :]                 # (n, 7, 7)
    M[:, 0, 0] *= 1 - lh_arr * la_arr * RHO
    M[:, 0, 1] *= 1 + lh_arr * RHO
    M[:, 1, 0] *= 1 + la_arr * RHO
    M[:, 1, 1] *= 1 - RHO
    M /= M.sum(axis=(1, 2), keepdims=True)
    tri = np.tril(np.ones((max_goals, max_goals)), -1)
    p_home = (M * tri[None]).sum(axis=(1, 2))
    p_draw = np.trace(M, axis1=1, axis2=2)
    p_away = (M * tri.T[None]).sum(axis=(1, 2))
    P = np.column_stack([p_home, p_draw, p_away])
    # temperature（与 predict_1x2 相同：log 域缩放再归一）
    logp = np.log(np.clip(P, 1e-12, None)) / TEMP
    P = np.exp(logp)
    return P / P.sum(axis=1, keepdims=True)


def rps_per_match(P, y):
    cp = np.cumsum(P, axis=1)[:, :2]
    oh = np.zeros_like(P); oh[np.arange(len(y)), y] = 1.0
    ca = np.cumsum(oh, axis=1)[:, :2]
    return ((cp - ca) ** 2).sum(axis=1) / 2.0


def main():
    df = pd.read_csv(FEATURES, low_memory=False)
    all_lh, all_la, all_y, all_season = [], [], [], []

    print("训练（forward-chain，1 seed —— γ 变换之间无训练噪声）")
    for ts in TEST_SEASONS:
        tr = df[(df["season"] >= "2016-2017") & (df["season"] < ts)]
        prev = sorted(tr["season"].unique())[-1]
        model = LGBMPoissonModel(rho=RHO, temperature=TEMP,
                                 params=LGB_PARAMS.copy(),
                                 feature_list=list(FEATURE_COLS))
        model.fit(tr[tr["season"] < prev], val_df=tr[tr["season"] == prev])
        te = df[df["season"] == ts]
        lh, la = model.predict_lambdas(te)
        all_lh.append(lh); all_la.append(la)
        all_y.append(np.array([match_outcome(r.home_goals, r.away_goals)
                               for r in te.itertuples()]))
        all_season.append(np.full(len(te), ts))
        # 诊断：模型隐含 HFA vs 实际
        act_h, act_a = te["home_goals"].mean(), te["away_goals"].mean()
        print(f"  {ts}: 模型 λh/λa = {lh.mean():.3f}/{la.mean():.3f}"
              f" (比 {lh.mean()/la.mean():.3f}) | 实际进球 {act_h:.3f}/{act_a:.3f}"
              f" (比 {act_h/act_a:.3f}) | N={len(te)}")

    lh = np.concatenate(all_lh); la = np.concatenate(all_la)
    y = np.concatenate(all_y); season = np.concatenate(all_season)
    n = len(y)

    # 每季各自的隐含 HFA（log 空间）
    log_h = np.zeros(n)
    for ts in TEST_SEASONS:
        m = season == ts
        log_h[m] = np.mean(np.log(lh[m])) - np.mean(np.log(la[m]))

    print(f"\nγ 刻度中立化（N={n}，s=0 为基线，负 ΔRPS = 更好）")
    print(f"{'s':>6} {'RPS':>9} {'ΔRPS vs s=0':>26} {'命中率':>7} "
          f"{'平局pick':>8} {'平局pick命中':>10}")
    base_rps = rps_per_match(probs_from_lambdas(lh, la), y)
    for s in S_GRID:
        lh2 = lh * np.exp(-s / 2 * log_h)
        la2 = la * np.exp(+s / 2 * log_h)
        P = probs_from_lambdas(lh2, la2)
        r = rps_per_match(P, y)
        pick = P.argmax(1)
        acc = (pick == y).mean()
        nd = int((pick == 1).sum())
        dh = f"{(y[pick == 1] == 1).mean():>10.1%}" if nd else f"{'—':>10}"
        if s == 0.0:
            tag = "(基线)"
        else:
            d = r - base_rps
            se = d.std(ddof=1) / np.sqrt(n)
            tag = f"{d.mean():+.5f} ± {se:.5f} z={d.mean()/se:+.1f}"
        print(f"{s:>6.2f} {r.mean():>9.5f} {tag:>26} {acc:>7.1%} {nd:>8} {dh}")

    # 平局基础占比参照
    print(f"\n实际平局率 {np.mean(y == 1):.1%} | 完全中立(s=1)时平局成为 argmax 的场次见上表")


if __name__ == "__main__":
    main()
