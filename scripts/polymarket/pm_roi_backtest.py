"""ROI 回测 —— 交接文档 §6 第 3 步：把「PM 更准 + 更便宜」翻译成下注策略。

评估口径（预注册，防多重比较捞噪声）
------------------------------------
1. 固定注 1 单位/注；逐注净收益的均值 ± SE；N 注。
2. 交易成本：
   - PM 买入价 = 该腿**原始盘口价** + 半价差 0.005（spread_report.txt：
     0.10~0.90 区间绝对价差中位 = 1 tick = 0.01）。敏感性再跑 +0.010。
   - 博彩公司无价差成本，水位已含在赔率里（MaxC = 收盘各家最优价）。
3. 结算：90 分钟赛果（pm_outcome，与 PM 结算及 football-data FTR 同口径）。
4. 预注册变体共 ~14 个 → **|z|>3 才算信号，2<|z|<3 只算线索**。

策略
----
A. 模型打 PM：p_model(o) − PM买入价(o) > τ 就买。第 1 步已证模型比 PM 差，
   预期死；跑它是为了给「模型边际」定一个 ROI 口径下的锚。
B. A 的平局特化（平局校准是模型在世界杯上验证过的相对强项）。
C. PM 当真值打博彩公司：EV = p_pm_raw(o) × 赔率(o) − 1 > τ 就在博彩公司下注。
   变体：MaxC（全季最优价）/ PSC（Pinnacle 收盘，前半季）。
D. 盲注基线（sanity）：全场次三个方向各注一遍，PM 与 MaxC 各一次——
   应当 ≈ −成本 / −水位；如果盲注都能赚，说明口径有 bug。

用法
----
    .venv/bin/python scripts/polymarket/pm_roi_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "data/polymarket/full_compare_2526.csv"
OUT = ROOT / "scripts/polymarket/reports/roi_backtest_2526.txt"

HALF_SPREAD = 0.005


def summarize(tag, rets, out, extra=""):
    rets = np.asarray(rets, float)
    if len(rets) == 0:
        out.append(f"  {tag:<44} N=0")
        return
    mu, se = rets.mean(), rets.std(ddof=1) / np.sqrt(len(rets))
    z = mu / se if se > 0 else 0.0
    out.append(f"  {tag:<44} N={len(rets):>4}  ROI {mu:+7.2%} ± {se:.2%}  z={z:+.2f}{extra}")


def main():
    m = pd.read_csv(CSV)
    y = m["pm_outcome"].values                       # 0/1/2 = H/D/A
    n = len(m)
    out = ["=" * 78,
           "ROI 回测 — 25/26 五大联赛 1749 场（固定注 1 单位，逐注收益 ± SE）",
           "=" * 78,
           f"成本口径：PM 买入 = 原始腿价 + {HALF_SPREAD}（半价差）；博彩公司用赔率原价。",
           "预注册变体 ~14 个：|z|>3 才算信号。", ""]

    # 原始腿价（full_compare 存的是归一化价，×pm_sum 还原盘口价）
    pm_raw = m[["pm_h", "pm_d", "pm_a"]].values * m[["pm_sum"]].values
    model = m[["model_prob_h", "model_prob_d", "model_prob_a"]].values
    maxc = m[["maxc_h", "maxc_d", "maxc_a"]].values          # 小数赔率
    psc = m[["psc_h", "psc_d", "psc_a"]].values
    has_pin = m["has_pinnacle"].values.astype(bool)
    win = np.zeros((n, 3)); win[np.arange(n), y] = 1.0

    def pm_bet_returns(mask2d, spread=HALF_SPREAD):
        """在 PM 按 mask 买入，逐注净收益（买价 q，赢赔 1）。"""
        q = pm_raw + spread
        rets = (win / q - 1.0)[mask2d & (q < 1.0)]
        return rets

    def book_bet_returns(odds, mask2d):
        rets = (win * odds - 1.0)[mask2d & np.isfinite(odds)]
        return rets

    # ── D. 盲注基线 ─────────────────────────────────────────────────────
    out.append("D. 盲注基线（sanity，应≈−成本/−水位）")
    all_mask = np.ones((n, 3), bool)
    summarize("PM 全方向盲注 (+0.005)", pm_bet_returns(all_mask), out)
    summarize("MaxC 全方向盲注", book_bet_returns(maxc, all_mask), out)
    summarize("PSC 全方向盲注 (前半季)", book_bet_returns(psc, all_mask & has_pin[:, None]), out)

    # ── A. 模型打 PM ────────────────────────────────────────────────────
    out.append("\nA. 模型打 PM（p_model − PM买入价 > τ）")
    for tau in (0.02, 0.03, 0.05):
        mask = model - (pm_raw + HALF_SPREAD) > tau
        summarize(f"τ={tau:.2f}", pm_bet_returns(mask), out)

    # ── B. 平局特化 ─────────────────────────────────────────────────────
    out.append("\nB. 模型打 PM，只买平局")
    for tau in (0.02, 0.03, 0.05):
        mask = np.zeros((n, 3), bool)
        mask[:, 1] = model[:, 1] - (pm_raw[:, 1] + HALF_SPREAD) > tau
        summarize(f"τ={tau:.2f}", pm_bet_returns(mask), out)

    # ── C. PM 当真值打博彩公司 ──────────────────────────────────────────
    out.append("\nC. PM 当真值打博彩公司（EV = p_pm_raw × 赔率 − 1 > τ）")
    ev_max = pm_raw * maxc - 1.0
    for tau in (0.00, 0.02, 0.05):
        summarize(f"MaxC 全季 τ={tau:.2f}", book_bet_returns(maxc, ev_max > tau), out)
    ev_psc = pm_raw * psc - 1.0
    for tau in (0.00, 0.02):
        summarize(f"PSC 前半季 τ={tau:.2f}",
                  book_bet_returns(psc, (ev_psc > tau) & has_pin[:, None]), out)

    # C 的方向拆分（τ=0.02, MaxC）：平局是否贡献主要边际
    out.append("\n  C(MaxC, τ=0.02) 按方向拆分：")
    for i, name in enumerate(["主胜", "平局", "客胜"]):
        mask = np.zeros((n, 3), bool)
        mask[:, i] = ev_max[:, i] > 0.02
        summarize(f"  只买{name}", book_bet_returns(maxc, mask), out)

    # 敏感性：PM 成本翻倍
    out.append("\n敏感性：PM 半价差 0.005 → 0.010（A, τ=0.03）")
    mask = model - (pm_raw + 0.010) > 0.03
    summarize("τ=0.03, spread=0.010", pm_bet_returns(mask, spread=0.010), out)

    report = "\n".join(out)
    OUT.write_text(report + "\n")
    print(report)
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    main()
