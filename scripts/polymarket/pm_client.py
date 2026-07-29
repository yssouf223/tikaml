"""Polymarket 只读 HTTP 客户端（纯 stdlib，无第三方依赖）。

实测结论（2026-07-28，复跑 `python probe_api.py` 可复核）：
  - CLOB 的所有深度端点（/book /books /midpoint(s) /spread(s) /price(s)
    /last-trade-price(s) /tick-size /prices-history）**均无需认证**
  - /trades 需要 L2 认证，返回 401
  - POST /books 可一次批量取 >=500 个 token 的完整挂单簿
  - 未观察到 429；瓶颈是网络延迟（keep-alive 约 390ms/次）
"""
from __future__ import annotations

import http.client
import json
import random
import ssl
import threading
import time
import urllib.parse

CLOB_HOST = "clob.polymarket.com"
GAMMA_HOST = "gamma-api.polymarket.com"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
BASE_HEADERS = {
    "User-Agent": UA,          # 不带 UA 会被 Cloudflare 403
    "Accept": "application/json",
    "Accept-Encoding": "identity",
}


class RateLimiter:
    """全局最小间隔限流（进程内，线程安全）。"""

    def __init__(self, min_interval: float = 0.25):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            gap = self.min_interval - (now - self._last)
            if gap > 0:
                time.sleep(gap)
                now = time.monotonic()
            self._last = now


class PMError(RuntimeError):
    def __init__(self, status, body, url):
        super().__init__(f"HTTP {status} {url}: {body[:200]}")
        self.status = status
        self.body = body
        self.url = url


class HttpClient:
    """带连接复用 + 退避重试的极简 HTTPS 客户端。

    keep-alive 实测把单次延迟从 ~750ms 降到 ~390ms，长期采集值得。
    服务端偶尔回 `Connection: close`，这里遇到断连自动重连。
    """

    def __init__(self, host: str, limiter: RateLimiter, timeout: float = 25.0,
                 max_retries: int = 3, log=None):
        self.host = host
        self.limiter = limiter
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = log or (lambda *a, **k: None)
        self._ctx = ssl.create_default_context()
        self._conn = None
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "retries": 0, "errors": 0, "bytes": 0}

    def _connect(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = http.client.HTTPSConnection(
            self.host, timeout=self.timeout, context=self._ctx
        )

    def request(self, method: str, path: str, body=None, extra_headers=None):
        headers = dict(BASE_HEADERS)
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        if extra_headers:
            headers.update(extra_headers)

        last_exc = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                # 指数退避 + 抖动；429 额外加重
                backoff = min(30.0, (2.0 ** attempt)) * (0.75 + 0.5 * random.random())
                if isinstance(last_exc, PMError) and last_exc.status == 429:
                    backoff *= 3
                self.log("retry", path=path[:60], attempt=attempt,
                         sleep=round(backoff, 2), why=str(last_exc)[:120])
                time.sleep(backoff)
                self.stats["retries"] += 1
            try:
                self.limiter.wait()
                with self._lock:
                    if self._conn is None:
                        self._connect()
                    try:
                        self._conn.request(method, path, body=payload, headers=headers)
                        resp = self._conn.getresponse()
                        raw = resp.read()
                    except (http.client.HTTPException, OSError):
                        # 连接被服务端关掉了，重连再试一次（不计入 retry 次数上限）
                        self._connect()
                        self._conn.request(method, path, body=payload, headers=headers)
                        resp = self._conn.getresponse()
                        raw = resp.read()
                    if resp.getheader("Connection", "").lower() == "close":
                        self._connect()
                self.stats["requests"] += 1
                self.stats["bytes"] += len(raw)
                if resp.status >= 400:
                    raise PMError(resp.status, raw.decode("utf-8", "replace"), path)
                return json.loads(raw) if raw else None
            except PMError as e:
                last_exc = e
                self.stats["errors"] += 1
                # 4xx（除 429）重试无意义，直接抛
                if e.status not in (429, 500, 502, 503, 504, 520, 521, 522, 524):
                    raise
            except Exception as e:  # noqa: BLE001 - 网络层任何异常都退避重试
                last_exc = e
                self.stats["errors"] += 1
                with self._lock:
                    self._connect()
        raise last_exc if last_exc else RuntimeError("unreachable")

    def get(self, path, params=None):
        if params:
            path = path + "?" + urllib.parse.urlencode(params)
        return self.request("GET", path)

    def post(self, path, body):
        return self.request("POST", path, body=body)


class PolymarketRO:
    """只读接口封装。CLOB 与 Gamma 各持一条连接，共用一个限流器。"""

    def __init__(self, min_interval=0.25, log=None):
        self.limiter = RateLimiter(min_interval)
        self.clob = HttpClient(CLOB_HOST, self.limiter, log=log)
        self.gamma = HttpClient(GAMMA_HOST, self.limiter, log=log)

    # ---------- CLOB ----------
    def book(self, token_id: str):
        return self.clob.get("/book", {"token_id": token_id})

    def books(self, token_ids):
        """批量挂单簿。实测 500 个 token 单请求可行（~1.9s / 1.4MB）。
        注意：服务端可能**静默丢弃**个别 token，调用方需自行核对补齐。"""
        return self.clob.post("/books", [{"token_id": t} for t in token_ids])

    def midpoints(self, token_ids):
        return self.clob.post("/midpoints", [{"token_id": t} for t in token_ids])

    def spreads(self, token_ids):
        return self.clob.post("/spreads", [{"token_id": t} for t in token_ids])

    def prices(self, token_ids):
        body = []
        for t in token_ids:
            body.append({"token_id": t, "side": "BUY"})
            body.append({"token_id": t, "side": "SELL"})
        return self.clob.post("/prices", body)

    def tick_size(self, token_id):
        return self.clob.get("/tick-size", {"token_id": token_id})

    def prices_history(self, token_id, interval="1d", fidelity=1):
        return self.clob.get("/prices-history",
                             {"market": token_id, "interval": interval,
                              "fidelity": fidelity})

    # ---------- Gamma ----------
    def events(self, **params):
        params.setdefault("limit", 100)
        return self.gamma.get("/events", params)

    def markets(self, **params):
        params.setdefault("limit", 100)
        return self.gamma.get("/markets", params)

    def tag_by_slug(self, slug):
        return self.gamma.get(f"/tags/slug/{slug}")


# ---------------------------------------------------------------- 盘口解析

def parse_levels(raw_levels, side):
    """CLOB 返回的挂单排序是**反直觉**的，实测（2026-07-28）：
       bids  按价格**升序**  → 最优买价 = 最后一个 = max
       asks  按价格**降序**  → 最优卖价 = 最后一个 = min
    这里统一归一化成「从最优价开始」的列表。
    """
    lv = [(float(x["price"]), float(x["size"])) for x in (raw_levels or [])]
    if side == "bid":
        lv.sort(key=lambda x: -x[0])   # 买：价高者优先
    else:
        lv.sort(key=lambda x: x[0])    # 卖：价低者优先
    return lv


def walk_cost(levels, notional_usd):
    """按名义金额吃单，返回 (成交均价 VWAP, 实际成交金额, 是否吃穿)。
    Polymarket 一份合约结算价 0 或 1，价格即概率，名义金额 = price*size。
    """
    remaining = notional_usd
    spent = 0.0
    shares = 0.0
    for price, size in levels:
        avail = price * size
        take = min(avail, remaining)
        if take <= 0:
            break
        spent += take
        shares += take / price
        remaining -= take
        if remaining <= 1e-9:
            break
    if shares <= 0:
        return None, 0.0, True
    return spent / shares, spent, remaining > 1e-9


def cum_notional(levels, best, max_dist):
    """距最优价 max_dist 以内的累计名义金额（USD）。"""
    tot = 0.0
    for price, size in levels:
        if abs(price - best) > max_dist + 1e-12:
            break
        tot += price * size
    return tot


def summarize_book(book, depth_n=10, walk_sizes=(100, 500, 1000, 5000)):
    """把一份原始 /book 压成一行结构化快照。"""
    bids = parse_levels(book.get("bids"), "bid")
    asks = parse_levels(book.get("asks"), "ask")
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    tick = float(book.get("tick_size") or 0.001)

    out = {
        "token_id": book.get("asset_id"),
        "condition_id": book.get("market"),
        "book_ts": int(book.get("timestamp") or 0),
        "book_hash": book.get("hash"),
        "tick_size": tick,
        "min_order_size": book.get("min_order_size"),
        "neg_risk": book.get("neg_risk"),
        "last_trade_price": book.get("last_trade_price"),
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "best_bid_size": bids[0][1] if bids else None,
        "best_ask_size": asks[0][1] if asks else None,
    }
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0
        out["spread"] = round(spread, 6)
        out["spread_ticks"] = round(spread / tick, 3)
        out["mid"] = round(mid, 6)
        # 相对价差：买一张的滑点成本占价格的比例（半价差 / 中价）
        out["half_spread_pct"] = round(100.0 * (spread / 2.0) / mid, 4) if mid > 0 else None
    else:
        out["spread"] = out["spread_ticks"] = out["mid"] = out["half_spread_pct"] = None

    out["bid_notional_total"] = round(sum(p * s for p, s in bids), 2)
    out["ask_notional_total"] = round(sum(p * s for p, s in asks), 2)
    if best_bid is not None:
        out["bid_notional_1c"] = round(cum_notional(bids, best_bid, 0.01), 2)
        out["bid_notional_5c"] = round(cum_notional(bids, best_bid, 0.05), 2)
    if best_ask is not None:
        out["ask_notional_1c"] = round(cum_notional(asks, best_ask, 0.01), 2)
        out["ask_notional_5c"] = round(cum_notional(asks, best_ask, 0.05), 2)

    out["bids"] = [[round(p, 6), round(s, 4)] for p, s in bids[:depth_n]]
    out["asks"] = [[round(p, 6), round(s, 4)] for p, s in asks[:depth_n]]

    # 吃单滑点：以 mid 为基准，买 N 美元名义要付多高的均价
    mid = out.get("mid")
    walk = {}
    for n in walk_sizes:
        vwap, filled, exhausted = walk_cost(asks, n)
        entry = {"vwap": round(vwap, 6) if vwap else None,
                 "filled": round(filled, 2), "exhausted": exhausted}
        if vwap and mid:
            entry["slip_pct_vs_mid"] = round(100.0 * (vwap - mid) / mid, 4)
        walk[str(n)] = entry
    out["buy_walk"] = walk
    return out
