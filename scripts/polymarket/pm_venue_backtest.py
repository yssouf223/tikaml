"""只在 Polymarket 下注的策略回测 —— 用户约束：无法去博彩公司下注。

信号源只能是：模型预测、博彩公司公开赔率。执行场所只有 PM。

预注册策略（防多重比较；|z|>3 才算信号）
--------------------------------------
E1. 模型置信分档：买模型 argmax，按模型最大概率分档（回应「置信区间成功率」
    的口径，给出 PM 场内的命中率与 ROI——命中率高 ≠ ROI 正）。
E2. 市场热门分档：买 PM 自己的热门，按其买入价分档（favorite-longshot 结构）。
E3. 收盘书商信号：AvgC 去水概率 − PM 买入价 > τ 时买入该腿。
    （混合测试 w*(PM)=0.95：书商只有 ~5% 增量信息，能不能盖过 1% 价差？）
E4. 赛前书商信号 → PM 早盘（本前提下最有希望的一条）：
    T−24h 时，书商赛前价（Avg，公开可得）去水概率 − PM 早盘买入价 > τ 就买。
    PM 早盘已证明比收盘钝 0.0015 RPS；书商赛前价发布于 T−72h 左右，
    信号在 T−24h 完全可得。**主指标 = 收盘估值的漂移捕获（SE 小），
    副指标 = 结算实收**。反向选注应捕获为负（对照）。

成本：买入 = 腿价 + 0.005（半价差中位）；敏感性 +0.010。结算 90 分钟。

用法
----
    .venv/bin/python scripts/polymarket/pm_venue_backtest.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "scripts/polymarket/reports/pm_venue_backtest_2526.txt"
HS = 0.005


def devig(df, p):
    inv = np.column_stack([1.0 / df[f"{p}_{k}"] for k in "hda"])
    return inv / inv.sum(axis=1, keepdims=True)


def summarize(tag, rets, out, unit="ROI"):
    rets = np.asarray(rets, float)
    rets = rets[np.isfinite(rets)]
    if len(rets) == 0:
        out.append(f"  {tag:<40} N=0")
        return
    mu, se = rets.mean(), rets.std(ddof=1) / np.sqrt(len(rets))
    z = mu / se if se > 0 else 0.0
    out.append(f"  {tag:<40} N={len(rets):>4}  {unit} {mu:+7.2%} ± {se:.2%}  z={z:+.2f}")


def main():
    m = pd.read_csv(ROOT / "data/polymarket/full_compare_2526.csv")
    tp = pd.read_csv(ROOT / "data/polymarket/timepoint_prices_2526.csv").drop(columns=["league"])
    m = m.merge(tp, on="slug")
    n = len(m)
    y = m["pm_outcome"].values
    win = np.zeros((n, 3)); win[np.arange(n), y] = 1.0

    pm_raw = m[["pm_h", "pm_d", "pm_a"]].values * m[["pm_sum"]].values  # 收盘腿价
    q24 = m[["q24_h", "q24_d", "q24_a"]].values                          # T−24h 腿价
    qc = m[["qc_h", "qc_d", "qc_a"]].values
    model = m[["model_prob_h", "model_prob_d", "model_prob_a"]].values
    avgc_p = devig(m, "avgc")      # 收盘书商去水概率
    avg_p = devig(m, "avg")        # 赛前书商去水概率（T−72h 左右发布）
    has_pin = m["has_pinnacle"].values.astype(bool)
    ps_p = np.where(has_pin[:, None], devig(m, "ps"), np.nan)

    out = ["=" * 78,
           "PM 场内策略回测 — 只在 Polymarket 下注（1749 场，买入=腿价+0.005）",
           "=" * 78, ""]

    def buy_ret(price2d, mask2d, spread=HS):
        q = price2d + spread
        ok = mask2d & (q < 1.0) & np.isfinite(q)
        return (win / q - 1.0)[ok]

    # ── E1 模型置信分档 ──────────────────────────────────────────────────
    out.append("E1. 买模型 argmax（PM 收盘价），按模型最大概率分档")
    am = model.argmax(1)
    pmax = model.max(1)
    sel_base = np.zeros((n, 3), bool); sel_base[np.arange(n), am] = True
    for lo, hi in [(0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
        band = (pmax >= lo) & (pmax < hi)
        mask = sel_base & band[:, None]
        hit = (am == y)[band].mean() if band.any() else np.nan
        summarize(f"p_max∈[{lo:.2f},{hi:.2f})  命中率 {hit:.1%}", buy_ret(pm_raw, mask), out)

    # ── E2 市场热门分档 ──────────────────────────────────────────────────
    out.append("\nE2. 买 PM 自己的热门（收盘），按买入价分档")
    fam = pm_raw.argmax(1)
    fsel = np.zeros((n, 3), bool); fsel[np.arange(n), fam] = True
    fprice = pm_raw[np.arange(n), fam] + HS
    for lo, hi in [(0.0, 0.45), (0.45, 0.60), (0.60, 0.75), (0.75, 1.01)]:
        band = (fprice >= lo) & (fprice < hi)
        summarize(f"买入价∈[{lo:.2f},{hi:.2f})  命中率 "
                  f"{(fam == y)[band].mean():.1%}" if band.any() else "空",
                  buy_ret(pm_raw, fsel & band[:, None]), out)

    # ── E3 收盘书商信号 ──────────────────────────────────────────────────
    out.append("\nE3. 收盘：AvgC 去水概率 − PM 买入价 > τ 时买入")
    for tau in (0.01, 0.02, 0.03):
        mask = avgc_p - (pm_raw + HS) > tau
        summarize(f"τ={tau:.2f}", buy_ret(pm_raw, mask), out)

    # ── E4 赛前书商信号 → PM 早盘 ────────────────────────────────────────
    out.append("\nE4. T−24h：书商赛前价(Avg 去水) − PM 早盘买入价 > τ 时买入")
    out.append("    主指标=漂移捕获（收盘−买入，SE 小）；副指标=结算实收")
    for tau in (0.01, 0.02, 0.03):
        mask = avg_p - (q24 + HS) > tau
        drift = (qc - q24 - HS)[mask & np.isfinite(q24)]
        summarize(f"τ={tau:.2f} 漂移捕获", drift, out, unit="Δp ")
        summarize(f"τ={tau:.2f} 结算实收", buy_ret(q24, mask), out)
    out.append("  反向对照（书商比 PM 早盘更悲观 τ=0.02，应为负）:")
    mask = avg_p - (q24 + HS) < -0.02
    summarize("  反向 漂移捕获", (qc - q24 - HS)[mask & np.isfinite(q24)], out, unit="Δp ")

    out.append("\n  E4(τ=0.02) 用 Pinnacle 赛前价替代 Avg（前半季子集）:")
    mask = ps_p - (q24 + HS) > 0.02
    summarize("  PS 信号 漂移捕获", (qc - q24 - HS)[mask & np.isfinite(q24) & np.isfinite(ps_p)],
              out, unit="Δp ")
    summarize("  PS 信号 结算实收", buy_ret(q24, mask & np.isfinite(ps_p)), out)

    # 方向拆分（E4 τ=0.02, Avg）
    out.append("\n  E4(τ=0.02, Avg) 按方向拆分（漂移捕获）:")
    mask = avg_p - (q24 + HS) > 0.02
    for i, nm in enumerate(["主胜", "平局", "客胜"]):
        mm = np.zeros((n, 3), bool); mm[:, i] = mask[:, i]
        summarize(f"  只买{nm}", (qc - q24 - HS)[mm & np.isfinite(q24)], out, unit="Δp ")

    # ── 基线与敏感性 ─────────────────────────────────────────────────────
    out.append("\n基线与敏感性")
    summarize("盲注: T−24h 全腿买入（应≈−0.005）",
              (qc - q24 - HS)[np.isfinite(q24)], out, unit="Δp ")
    mask = avg_p - (q24 + 0.010) > 0.02
    summarize("E4 τ=0.02, 半价差=0.010 漂移捕获",
              (qc - q24 - 0.010)[mask & np.isfinite(q24)], out, unit="Δp ")
    summarize("E4 τ=0.02, 半价差=0.010 结算实收", buy_ret(q24, mask, spread=0.010), out)

    report = "\n".join(out)
    OUT.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
