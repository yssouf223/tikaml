#!/usr/bin/env python3
"""买卖价差 / 滑点量化。

回答的问题：我们之前算 ROI 用的是「最后成交价」，但真实下单要吃卖一。
这中间差多少？对 1% 量级的 edge 来说，这个差是不是致命的？

做法：把当前**全部未结算**市场的挂单簿一次性抓下来（POST /books 批量），
按价格档、tick size、盘口类型（比赛盘 / 期货盘）、成交额分组统计：
  - 绝对价差 best_ask - best_bid
  - 相对成本 (best_ask - mid) / mid       ← 半价差，买一手的最小滑点
  - 相对成本 (best_ask - last_trade)/last_trade  ← 我们回测口径的实际偏差
  - 按 $100 / $500 / $1000 / $5000 名义金额吃单的 VWAP 滑点
"""
from __future__ import annotations

import json
import os
import re
import statistics as stt
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pm_client import PolymarketRO, summarize_book  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "spread_study.ndjson")

SPORT_PREFIX = re.compile(r"^(mlb|nba|nfl|nhl|mls|epl|ucl|atp|wta|itf|wnba|ufc|lol|cs2|dota2|val|"
                          r"bra\d|arg|mex|col|ecu\d|lig|ncaa)[-_]")


def pctl(vals, q):
    v = sorted(vals)
    if not v:
        return None
    return v[min(len(v) - 1, int(q * (len(v) - 1)))]


def desc(vals, fmt="{:.4f}"):
    if not vals:
        return "n=0"
    return (f"n={len(vals):<5} 中位={fmt.format(stt.median(vals))} "
            f"p25={fmt.format(pctl(vals,.25))} p75={fmt.format(pctl(vals,.75))} "
            f"p90={fmt.format(pctl(vals,.90))} 均值={fmt.format(sum(vals)/len(vals))}")


def collect(api, markets, chunk=200):
    """把 market 元数据展开成 token，批量取 book。"""
    tok_meta = {}
    for m in markets:
        try:
            toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) \
                else m.get("clobTokenIds")
            outs = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) \
                else (m.get("outcomes") or [])
        except Exception:  # noqa: BLE001
            continue
        if not toks:
            continue
        for i, t in enumerate(toks):
            tok_meta[t] = {
                "slug": m.get("slug"), "vol24": m.get("volume24hr") or 0,
                "liq": m.get("liquidityNum") or 0,
                "gameStart": m.get("gameStartTime"),
                "gamma_tick": m.get("orderPriceMinTickSize"),
                "outcome": outs[i] if i < len(outs) else f"idx{i}",
                "sportsMarketType": m.get("sportsMarketType"),
            }
    toks = list(tok_meta)
    print(f"待取 book 的 token 数: {len(toks)}")
    rows = []
    for i in range(0, len(toks), chunk):
        part = toks[i:i + chunk]
        try:
            books = api.books(part)
        except Exception as e:  # noqa: BLE001
            print("  批次失败", i, str(e)[:120])
            continue
        for b in books or []:
            if not b or not b.get("asset_id"):
                continue
            s = summarize_book(b)
            s.update(tok_meta.get(b["asset_id"], {}))
            rows.append(s)
        print(f"  {i+len(part)}/{len(toks)} 累计 {len(rows)}", flush=True)
    return rows


def report(rows):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"\n原始快照已写入 {OUT}（{len(rows)} 行）")

    two = [r for r in rows if r.get("spread") is not None]
    print(f"\n总 token {len(rows)}，双边有报价 {len(two)}，"
          f"单边/空盘 {len(rows)-len(two)}")

    def bucket(r):
        m = r["mid"]
        if m < 0.05:
            return "1. <0.05"
        if m < 0.10:
            return "2. 0.05-0.10"
        if m < 0.25:
            return "3. 0.10-0.25"
        if m < 0.45:
            return "4. 0.25-0.45"
        if m < 0.55:
            return "5. 0.45-0.55"
        if m < 0.75:
            return "6. 0.55-0.75"
        if m < 0.90:
            return "7. 0.75-0.90"
        if m < 0.95:
            return "8. 0.90-0.95"
        return "9. >0.95"

    print("\n" + "=" * 96)
    print("A. tick_size 与价格的关系（决定价差下限）")
    print("=" * 96)
    tb = defaultdict(Counter)
    for r in two:
        tb[bucket(r)][r["tick_size"]] += 1
    print(f"{'价格区间':<16}{'tick=0.001':>12}{'tick=0.01':>12}   最小可能价差占中价")
    for b in sorted(tb):
        c = tb[b]
        dom = 0.01 if c[0.01] >= c[0.001] else 0.001
        mids = [r["mid"] for r in two if bucket(r) == b]
        mm = stt.median(mids)
        print(f"{b:<16}{c[0.001]:>12}{c[0.01]:>12}   {100*dom/mm:>6.2f}%  (tick={dom}, 中价≈{mm:.3f})")

    print("\n" + "=" * 96)
    print("B. 实际买卖价差，按价格区间")
    print("=" * 96)
    for b in sorted(set(bucket(r) for r in two)):
        sub = [r for r in two if bucket(r) == b]
        print(f"\n[{b}]  n={len(sub)}")
        print("  绝对价差      " + desc([r["spread"] for r in sub]))
        print("  价差/tick     " + desc([r["spread_ticks"] for r in sub], "{:.1f}"))
        print("  半价差占中价  " + desc([r["half_spread_pct"] for r in sub
                                        if r.get("half_spread_pct") is not None], "{:.3f}%"))

    print("\n" + "=" * 96)
    print("C. 只看比赛盘（有 gameStartTime，与足球比赛盘同类），按流动性分层")
    print("=" * 96)
    games = [r for r in two if r.get("gameStart")]
    print(f"比赛盘 token 数 {len(games)}")
    for lo, hi, name in [(0, 1000, "vol24 <$1k 冷门"), (1000, 20000, "$1k-20k"),
                         (20000, 100000, "$20k-100k"), (100000, 1e12, ">$100k 热门")]:
        sub = [r for r in games if lo <= r["vol24"] < hi]
        if not sub:
            continue
        mid_only = [r for r in sub if 0.15 <= r["mid"] <= 0.85]
        print(f"\n[{name}] n={len(sub)}（其中中价 0.15-0.85 的 {len(mid_only)} 个）")
        print("  绝对价差      " + desc([r["spread"] for r in sub]))
        if mid_only:
            print("  半价差占中价(仅0.15-0.85)  "
                  + desc([r["half_spread_pct"] for r in mid_only], "{:.3f}%"))
            print("  卖一档名义金额 $ " + desc([r.get("ask_notional_1c") or 0
                                                for r in mid_only], "{:,.0f}"))

    print("\n" + "=" * 96)
    print("D. 吃单滑点：买入 N 美元名义，成交均价比中价贵多少（%）")
    print("=" * 96)
    live = [r for r in two if 0.15 <= r["mid"] <= 0.85]
    liq = [r for r in live if r["vol24"] >= 10000]
    for label, pool in [("全部中价盘(0.15-0.85)", live), ("其中 vol24>=$10k", liq)]:
        print(f"\n[{label}] n={len(pool)}")
        for n in ("100", "500", "1000", "5000"):
            v = [r["buy_walk"][n]["slip_pct_vs_mid"] for r in pool
                 if r["buy_walk"][n].get("slip_pct_vs_mid") is not None]
            ex = sum(1 for r in pool if r["buy_walk"][n]["exhausted"])
            print(f"  ${n:>5}  " + desc(v, "{:.3f}%") + f"  吃穿全簿 {ex} 个")

    print("\n" + "=" * 96)
    print("E. 回测口径偏差：best_ask 相对 last_trade_price 高多少")
    print("=" * 96)
    for label, pool in [("全部中价盘", live), ("比赛盘中价", [r for r in live if r.get("gameStart")])]:
        d = []
        for r in pool:
            lt = r.get("last_trade_price")
            try:
                lt = float(lt)   # CLOB 有时回字符串
            except (TypeError, ValueError):
                continue
            if 0 < lt < 1 and r.get("best_ask"):
                d.append(100.0 * (r["best_ask"] - lt) / lt)
        print(f"[{label}] " + desc(d, "{:+.3f}%"))

    print("\n" + "=" * 96)
    print("G. 【最贴近足球比赛盘的场景】比赛盘 + 中价 0.30-0.70 + vol24>=$20k")
    print("=" * 96)
    foot = [r for r in games if 0.30 <= r["mid"] <= 0.70 and r["vol24"] >= 20000]
    print(f"n={len(foot)}")
    if foot:
        print("  绝对价差        " + desc([r["spread"] for r in foot]))
        print("  半价差占中价    " + desc([r["half_spread_pct"] for r in foot], "{:.3f}%"))
        print("  卖一档名义 $    " + desc([r.get("ask_notional_1c") or 0 for r in foot], "{:,.0f}"))
        for n in ("100", "500", "1000"):
            v = [r["buy_walk"][n]["slip_pct_vs_mid"] for r in foot
                 if r["buy_walk"][n].get("slip_pct_vs_mid") is not None]
            print(f"  买 ${n:>5} 滑点  " + desc(v, "{:.3f}%"))
        share_1tick = sum(1 for r in foot if r["spread_ticks"] <= 1.01) / len(foot)
        share_le2 = sum(1 for r in foot if r["spread_ticks"] <= 2.01) / len(foot)
        print(f"  价差 = 1 个 tick 的占比: {share_1tick:.0%}；<=2 个 tick: {share_le2:.0%}")

    print("\n" + "=" * 96)
    print("I. 赛前 vs 赛后（关键切分：下注只会发生在开球前）")
    print("=" * 96)
    import datetime as _dt

    def _gs(r):
        s = r.get("gameStart")
        if not s:
            return None
        s = s.strip().replace(" ", "T")
        if s.endswith("+00"):
            s = s[:-3] + "+00:00"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            d = _dt.datetime.fromisoformat(s)
            return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            return None

    now = _dt.datetime.now(_dt.timezone.utc)
    pre, post = [], []
    for r in games:
        g = _gs(r)
        if g is None:
            continue
        (pre if g > now else post).append(r)
    for label, pool in [("赛前(未开球)", pre), ("已开球/已结束", post)]:
        core = [r for r in pool if 0.20 <= r["mid"] <= 0.80]
        print(f"\n[{label}] 全部 {len(pool)}，中价 0.20-0.80 的 {len(core)}")
        if core:
            print("  绝对价差      " + desc([r["spread"] for r in core]))
            print("  半价差占中价  " + desc([r["half_spread_pct"] for r in core], "{:.3f}%"))
            print("  卖一档名义 $  " + desc([r.get("ask_notional_1c") or 0 for r in core], "{:,.0f}"))
            v = [r["buy_walk"]["500"]["slip_pct_vs_mid"] for r in core
                 if r["buy_walk"]["500"].get("slip_pct_vs_mid") is not None]
            print("  买 $500 滑点  " + desc(v, "{:.3f}%"))
    # 赛前按距开球时间分层——这正是采集器要捕捉的规律
    print("\n  赛前按距开球时间分层（中价 0.20-0.80）:")
    for lo, hi, name in [(0, 2, "T-2h 内"), (2, 12, "T-2h~12h"),
                         (12, 48, "T-12h~48h"), (48, 1e9, "T+48h 以上")]:
        sub = [r for r in pre if 0.20 <= r["mid"] <= 0.80
               and lo <= (_gs(r) - now).total_seconds() / 3600 < hi]
        if sub:
            print(f"    {name:<12} n={len(sub):<4} 半价差中位="
                  f"{stt.median([r['half_spread_pct'] for r in sub]):.2f}%"
                  f"  卖一档名义中位=${stt.median([r.get('ask_notional_1c') or 0 for r in sub]):,.0f}")

    print("\n" + "=" * 96)
    print("H. 结论换算：往返成本 vs edge")
    print("=" * 96)
    core2 = [r for r in games if 0.30 <= r["mid"] <= 0.70]
    if core2:
        hs = sorted(r["half_spread_pct"] for r in core2)
        med = stt.median(hs)
        print(f"比赛盘中价 0.30-0.70 共 {len(core2)} 个，半价差中位 {med:.2f}%。")
        print(f"若持有到结算（只吃一次卖价，不用平仓），单边成本 ≈ {med:.2f}%；")
        print(f"若中途平仓（吃卖价买入 + 吃买价卖出），往返 ≈ {2*med:.2f}%。")

    print("\n" + "=" * 96)
    print("F. 最差 / 最好的比赛盘（中价 0.3-0.7，按半价差排序）")
    print("=" * 96)
    core = [r for r in games if 0.3 <= r["mid"] <= 0.7]
    core.sort(key=lambda r: r["half_spread_pct"])
    print("--- 最紧的 8 个 ---")
    for r in core[:8]:
        print(f"  {r['slug'][:46]:<46} {str(r['outcome'])[:16]:<16} "
              f"mid={r['mid']:.3f} spread={r['spread']:.3f} "
              f"半价差={r['half_spread_pct']:.2f}% vol24=${r['vol24']:,.0f}")
    print("--- 最宽的 8 个 ---")
    for r in core[-8:]:
        print(f"  {r['slug'][:46]:<46} {str(r['outcome'])[:16]:<16} "
              f"mid={r['mid']:.3f} spread={r['spread']:.3f} "
              f"半价差={r['half_spread_pct']:.2f}% vol24=${r['vol24']:,.0f}")


def main():
    if "--report-only" in sys.argv:
        rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
        print(f"复用已抓取的 {len(rows)} 条盘口快照（不再打扰 API）")
        report(rows)
        return
    api = PolymarketRO(min_interval=0.5)   # 与采集器并行跑，放缓一点
    cache = os.path.join(BASE, "all_active_markets_full.json")
    if os.path.exists(cache) and "--refresh" not in sys.argv:
        markets = json.load(open(cache, encoding="utf-8"))
        print(f"复用缓存 {cache}: {len(markets)} 个市场")
    else:
        markets = []
        for page in range(14):
            ms = api.markets(closed="false", active="true", limit=100,
                             offset=page * 100, order="volume24hr", ascending="false")
            if not ms:
                break
            markets.extend(ms)
            print(f"  gamma page {page+1}: 累计 {len(markets)}", flush=True)
        json.dump(markets, open(cache, "w"))
    markets = [m for m in markets if m.get("enableOrderBook", True)
               and m.get("acceptingOrders") is not False]
    print(f"可下单市场 {len(markets)}")
    rows = collect(api, markets)
    report(rows)


if __name__ == "__main__":
    main()
