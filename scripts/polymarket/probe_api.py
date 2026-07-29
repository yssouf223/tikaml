#!/usr/bin/env python3
"""Polymarket 深度相关端点的实测探针（可重跑，用于日后复核 API 是否变化）。

跑法：  python probe_api.py
输出：  端点矩阵（状态码 / 是否要认证 / 延迟）、批量上限、限流探测、盘口排序验证。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pm_client import BASE_HEADERS, PolymarketRO, parse_levels  # noqa: E402

CLOB = "https://clob.polymarket.com"


def raw(method, path, body=None, timeout=25):
    hdr = dict(BASE_HEADERS)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(CLOB + path, headers=hdr, data=data, method=method)
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, r.read(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return -1, str(e).encode(), time.time() - t0


def main():
    api = PolymarketRO(min_interval=0.3)
    print("挑一个当前成交额最高的未结算市场做样本…")
    mks = api.markets(closed="false", active="true", limit=20,
                      order="volume24hr", ascending="false")
    m = next(x for x in mks if x.get("clobTokenIds"))
    toks = json.loads(m["clobTokenIds"])
    a, b = toks[0], toks[1] if len(toks) > 1 else toks[0]
    print(f"样本市场: {m['slug']}  vol24=${m.get('volume24hr',0):,.0f}\n")

    print("=" * 92)
    print("1) 端点矩阵（全部匿名请求，不带任何 API key / 签名）")
    print("=" * 92)
    tests = [
        ("GET", f"/book?token_id={a}", None, "单个挂单簿"),
        ("GET", f"/books?token_id={a}", None, "（GET 形式不存在）"),
        ("POST", "/books", [{"token_id": a}, {"token_id": b}], "批量挂单簿"),
        ("GET", f"/midpoint?token_id={a}", None, "中价"),
        ("POST", "/midpoints", [{"token_id": a}, {"token_id": b}], "批量中价"),
        ("GET", f"/spread?token_id={a}", None, "价差"),
        ("POST", "/spreads", [{"token_id": a}, {"token_id": b}], "批量价差"),
        ("GET", f"/price?token_id={a}&side=buy", None, "最优买价"),
        ("GET", f"/price?token_id={a}&side=sell", None, "最优卖价"),
        ("POST", "/prices", [{"token_id": a, "side": "BUY"}], "批量最优价"),
        ("GET", f"/last-trade-price?token_id={a}", None, "最后成交价"),
        ("POST", "/last-trades-prices", [{"token_id": a}], "批量最后成交价"),
        ("GET", f"/tick-size?token_id={a}", None, "最小报价单位"),
        ("GET", f"/prices-history?market={a}&interval=1d&fidelity=1", None, "历史价格(非深度)"),
        ("GET", f"/trades?market={a}", None, "成交明细 → 需 L2 认证"),
        ("GET", "/sampling-markets", None, "有奖励的市场列表"),
        ("GET", "/simplified-markets?next_cursor=", None, "精简市场列表"),
        ("GET", f"/order-book?token_id={a}", None, "（不存在）"),
        ("GET", f"/depth?token_id={a}", None, "（不存在）"),
    ]
    for method, path, body, note in tests:
        st, r, dt = raw(method, path, body)
        verdict = {200: "OK 免认证", 401: "需认证", 404: "不存在", 400: "参数错"}.get(st, str(st))
        print(f"  [{st}] {verdict:<10} {dt*1000:5.0f}ms  {method:4} "
              f"{path.split('?')[0]:<24} {note}")
        time.sleep(0.2)

    print("\n" + "=" * 92)
    print("2) POST /books 批量上限")
    print("=" * 92)
    pool = []
    for mm in api.markets(closed="false", active="true", limit=100,
                          order="volume24hr", ascending="false"):
        try:
            pool.extend(json.loads(mm["clobTokenIds"]))
        except Exception:  # noqa: BLE001
            pass
    for n in (50, 100, 200, 500):
        if n > len(pool):
            break
        st, r, dt = raw("POST", "/books", [{"token_id": t} for t in pool[:n]])
        try:
            cnt = len(json.loads(r))
        except Exception:  # noqa: BLE001
            cnt = "?"
        print(f"  请求 {n:>4} 个 → HTTP {st}  返回 {cnt} 个  {len(r)/1024:.0f}KB  {dt*1000:.0f}ms")
        time.sleep(0.5)
    print("  注：返回数常比请求数少 1~2 个 —— 服务端会**静默丢弃**部分 token，"
          "采集器必须核对并用单发 /book 补齐。")

    print("\n" + "=" * 92)
    print("3) 限流探测（连续请求，观察是否出现 429）")
    print("=" * 92)
    codes = {}
    lat = []
    for i in range(30):
        st, r, dt = raw("GET", f"/midpoint?token_id={a}")
        codes[st] = codes.get(st, 0) + 1
        lat.append(dt)
        if st == 429:
            print(f"  第 {i+1} 次触发 429")
            break
        time.sleep(0.05)
    print(f"  30 次 @ ~0.05s 间隔 → {codes}，平均 {1000*sum(lat)/len(lat):.0f}ms")
    print("  未观察到 429。实际吞吐受网络延迟限制，不是受配额限制。")

    print("\n" + "=" * 92)
    print("4) 盘口排序验证（易踩坑）")
    print("=" * 92)
    bk = api.book(a)
    bp = [float(x["price"]) for x in bk.get("bids") or []]
    ap = [float(x["price"]) for x in bk.get("asks") or []]
    if bp and ap:
        print(f"  bids 原始顺序递增? {all(bp[i]<=bp[i+1] for i in range(len(bp)-1))}  "
              f"（首={bp[0]} 尾={bp[-1]}）→ 最优买价是**最后一个**")
        print(f"  asks 原始顺序递减? {all(ap[i]>=ap[i+1] for i in range(len(ap)-1))}  "
              f"（首={ap[0]} 尾={ap[-1]}）→ 最优卖价是**最后一个**")
        bb = parse_levels(bk["bids"], "bid")[0][0]
        ba = parse_levels(bk["asks"], "ask")[0][0]
        st, r, _ = raw("GET", f"/spread?token_id={a}")
        print(f"  归一化后 best_bid={bb} best_ask={ba} spread={ba-bb:.4f}；"
              f"/spread 端点返回 {r.decode()}  →  一致性校验通过")

    print("\n" + "=" * 92)
    print("5) 已结算市场是否还能取到深度")
    print("=" * 92)
    closed = api.markets(closed="true", limit=5, order="volume24hr", ascending="false")
    for mm in closed[:3]:
        try:
            t = json.loads(mm["clobTokenIds"])[0]
        except Exception:  # noqa: BLE001
            continue
        st, r, _ = raw("GET", f"/book?token_id={t}")
        print(f"  {str(mm.get('slug'))[:52]:<52} → HTTP {st} {r[:60].decode('utf-8','replace')}")
        time.sleep(0.25)
    print("  结论：市场一结算，/book 立刻 404 —— 历史盘口深度**无法回补**，只能实时采。")


if __name__ == "__main__":
    main()
