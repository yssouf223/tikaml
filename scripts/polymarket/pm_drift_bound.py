"""收盘漂移的可捕获上界 —— 第 3 步的 kill-or-continue 闸门。

背景（交接文档 §2.3）
--------------------
上一轮唯一没被证伪的信号：收盘漂移方向可预测（相关 0.16–0.29），但以 RPS 计
投入产出不成立。ROI 视角下重问：**假设完美预见收盘价**，在 T−Δ 买入将要涨的
那条腿、扣掉半价差后，每场还能剩多少？这是任何漂移策略的硬上界——
上界不为正，这条线直接判死；为正，再按 10~30% 的现实捕获率折算是否值得做。

方法
----
1. 解码全部 5265 个二元市场的分钟级序列（p×10000 整数 + dt 差分编码）。
2. 对 Δ ∈ {24h, 6h, 1h}：取 T−Δ 时刻的最近价 q_t 与收盘价 q_T。
3. 假设收盘价 ≈ 真实概率（第 1 步已证收盘价是当时最准的公开估计），
   完美预见者在每场三条腿里选 (q_T − q_t) 最大的一条买入：
   净捕获 = q_T − q_t − 半价差(0.005)。
4. 报告：净捕获的均值/中位/为正比例；|移动|>价差的腿占比；
   以及市场在最后 24h 的 RPS 增益（信息到底是什么时候进价的）。

用法
----
    .venv/bin/python scripts/polymarket/pm_drift_bound.py
"""

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SERIES = ROOT / "data/polymarket/series"
OUT = ROOT / "scripts/polymarket/reports/drift_bound_2526.txt"

HALF_SPREAD = 0.005
LAGS = {"24h": 86400, "6h": 21600, "1h": 3600}
LEGS = ("home", "draw", "away")

# README_format.md 列出的改期重复（保留长序列）与升降级附加赛
DROP = {"lal-val-ovi-2025-09-30", "fl1-olm-psg-2025-09-22",
        "bun-pad-wol-2026-05-25", "fl1-se-nic-2026-05-26", "fl1-nic-se-2026-05-29"}


def main():
    rows = []          # 每场: {lag: (q_t[3], q_T[3]), 'res': outcome}
    n_pts = 0
    for f in sorted(SERIES.glob("*_series.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if r["slug"] in DROP:
                continue
            kick = r["kick_ts"]
            qt = {}
            ok = True
            res = []
            for leg in LEGS:
                mk = r["mk"][leg]
                if "p" not in mk:
                    ok = False
                    break
                t = mk["t0"] + np.concatenate([[0], np.cumsum(mk["dt"])])
                p = np.asarray(mk["p"], float) / 10000.0
                n_pts += len(p)
                res.append(mk["res"])
                for lag, sec in LAGS.items():
                    i = np.searchsorted(t, kick - sec, side="right") - 1
                    qt.setdefault(lag, []).append(p[i] if i >= 0 else np.nan)
                qt.setdefault("close", []).append(p[-1])
            if not ok or sorted(res) != [0.0, 0.0, 1.0]:
                continue
            rows.append({"q": qt, "y": int(np.argmax(res)), "lg": r["league"]})

    out = ["=" * 78,
           f"收盘漂移可捕获上界 — {len(rows)} 场 / {n_pts / 1e6:.1f}M 价格点",
           "=" * 78,
           f"口径：完美预见收盘价，每场三腿选涨幅最大者在 T−Δ 买入，"
           f"净捕获 = q_T − q_t − {HALF_SPREAD}", ""]

    close = np.array([r["q"]["close"] for r in rows])
    y = np.array([r["y"] for r in rows])

    def rps(P):
        Pn = P / P.sum(axis=1, keepdims=True)
        cp = np.cumsum(Pn, axis=1)[:, :2]
        oh = np.zeros_like(Pn); oh[np.arange(len(y)), y] = 1.0
        ca = np.cumsum(oh, axis=1)[:, :2]
        return ((cp - ca) ** 2).sum(axis=1) / 2.0

    out.append("一、信息何时进价（各时点价格的 RPS，归一化后）")
    out.append(f"  {'收盘':<8} {rps(close).mean():.5f}")
    for lag in LAGS:
        q = np.array([r["q"][lag] for r in rows])
        valid = ~np.isnan(q).any(axis=1)
        d = rps(np.where(np.isnan(q), close, q))[valid] - rps(close)[valid]
        out.append(f"  T−{lag:<6} {rps(np.where(np.isnan(q), close, q))[valid].mean():.5f}"
                   f"  (vs 收盘 Δ {d.mean():+.5f} ± {d.std(ddof=1)/np.sqrt(valid.sum()):.5f},"
                   f" N={valid.sum()})")

    out.append("\n二、完美预见的净捕获（每场选最优腿，扣半价差）")
    for lag in LAGS:
        q = np.array([r["q"][lag] for r in rows])
        valid = ~np.isnan(q).any(axis=1)
        move = close[valid] - q[valid]                  # 各腿涨跌
        best = move.max(axis=1) - HALF_SPREAD           # 完美选腿净捕获
        pos = (best > 0).mean()
        out.append(f"  T−{lag:<4} 净捕获 均值 {best.mean():+.4f} | 中位 {np.median(best):+.4f}"
                   f" | >0 占比 {pos:.1%} | |单腿移动|>0.01 占比 "
                   f"{(np.abs(move) > 0.01).mean():.1%}")

    out.append("\n三、按联赛（T−24h 净捕获均值）")
    q24 = np.array([r["q"]["24h"] for r in rows])
    lgs = np.array([r["lg"] for r in rows])
    for lg in sorted(set(lgs)):
        sel = (lgs == lg) & ~np.isnan(q24).any(axis=1)
        best = (close[sel] - q24[sel]).max(axis=1) - HALF_SPREAD
        out.append(f"  {lg:<12} {best.mean():+.4f} (N={sel.sum()})")

    out.append("\n四、换算：现实捕获率折算的每注期望")
    q24v = ~np.isnan(q24).any(axis=1)
    best24 = (close[q24v] - q24[q24v]).max(axis=1) - HALF_SPREAD
    for frac in (1.0, 0.3, 0.1):
        out.append(f"  捕获率 {frac:.0%}: 每场 {best24.mean() * frac:+.4f}"
                   f"（以 0.4 左右的腿价计 ≈ ROI {best24.mean() * frac / 0.4:+.1%}）")

    report = "\n".join(out)
    OUT.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
