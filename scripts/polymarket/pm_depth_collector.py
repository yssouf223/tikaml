#!/usr/bin/env python3
"""Polymarket 盘口深度实时采集器（长期无人值守）。

为什么要有它：CLOB 的 /book 只对**未结算**市场返回数据，市场一结算就 404。
也就是说历史盘口深度**无法回补**，只能从现在开始不间断地采。8/15 新赛季开赛前
必须先把这个跑起来。

用法
----
  # 1) 看看会跟踪哪些市场（不落盘）
  python pm_depth_collector.py discover --tags mlb,atp,wta --horizon-hours 48

  # 2) 正式采集（Ctrl-C 可随时停，重启自动续上）
  python pm_depth_collector.py run --tags epl,mlb --horizon-hours 72

  # 3) 休赛期用「按成交额挑活跃市场」模式做压力测试
  python pm_depth_collector.py run --mode top-volume --top-n 40 --duration 600

  # 4) 分析已采数据的买卖价差
  python pm_depth_collector.py analyze

设计要点
--------
1. 采样节奏按距开球时间自适应：T-24h 每小时 → T-6h 每 15 分 → T-2h 每 5 分
   → T-15min 每 1 分 → 赛中每 2 分。开球时间取 Gamma 的 `gameStartTime`。
2. 批量：所有到期 token 汇总后走 POST /books（一次最多 250 个），
   把 N 个市场的请求数压成 1 个。服务端会静默丢 token，丢的用单发 /book 补。
3. 断点续传：每轮把 state.json 原子写盘（tmp + rename）。
   快照是 NDJSON 追加写 + fsync，进程被 kill 也最多丢最后一行。
4. 重试：pm_client.HttpClient 里三次指数退避；单个市场连续失败会被隔离，
   不拖累整轮。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import datetime as dt
from collections import defaultdict

# 足球主盘（1X2 各一个 Yes/No 二元盘）的 slug 形如
#   epl-liv-bou-2026-08-15-liv / mls-nyc-tor-2026-07-31-nyc / ucl-crv-lar-2026-07-29-lar
# 其余 corners-*/exact-score-*/total-*/spread-* 都是玩法盘，成交额通常为 0。
MONEYLINE_RE = r"^[a-z0-9]{2,6}-[a-z]{2,4}-[a-z]{2,4}-\d{4}-\d{2}-\d{2}(-[a-z]{2,4})?$"
_slug_filter = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pm_client import PolymarketRO, PMError, summarize_book  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
LOGS = os.path.join(BASE, "logs")
STATE_PATH = os.path.join(DATA, "state.json")
TRACK_PATH = os.path.join(DATA, "tracked.json")
SNAP_DIR = os.path.join(DATA, "snapshots")

BOOKS_CHUNK = 250        # 单次 POST /books 的 token 数（实测 500 也行，留余量）
MAX_CONSEC_FAIL = 8      # 单市场连续失败次数上限，超过则隔离

# 距开球时间 t_minus（正=还没开球）-> 采样间隔（秒）。
# 取第一条满足 t_minus <= 上界 的规则，所以列表必须按上界升序。
CADENCE = [
    (-7200,      120),   # 开球后超过 2 小时（比赛已结束，等结算）：2 分钟
    (900,         60),   # T-15min 一直到开球后 2 小时（含全场赛中）：1 分钟
    (7200,       300),   # T-2h ~ T-15min：5 分钟
    (21600,      900),   # T-6h ~ T-2h：15 分钟
    (86400,     3600),   # T-24h ~ T-6h：1 小时
    (float("inf"), 21600),  # T-24h 以前：6 小时（只维持存在性，省请求）
]
STALE_AFTER = 4 * 3600   # 开球后多久停止跟踪（秒）；足球 90min+补时+结算足够


def now_ms():
    return int(time.time() * 1000)


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, event, **kv):
        rec = {"ts": utcnow().isoformat(timespec="seconds"), "event": event}
        rec.update(kv)
        line = json.dumps(rec, ensure_ascii=False)
        self.f.write(line + "\n")
        self.f.flush()
        print(line, flush=True)


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_game_start(s):
    """Gamma 的 gameStartTime 形如 '2026-07-27 23:40:00+00'，非标准 ISO。"""
    if not s:
        return None
    s = s.strip().replace(" ", "T")
    if s.endswith("+00"):
        s = s[:-3] + "+00:00"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = dt.datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def cadence_for(t_minus_sec):
    """t_minus_sec = 开球时间 - 现在（正数=还没开球）。"""
    if t_minus_sec is None:
        return 900          # 没有开球时间的市场（期货盘）：固定 15 分钟
    for upper, interval in CADENCE:
        if t_minus_sec <= upper:
            return interval
    return 21600


# ------------------------------------------------------------------ 发现

def discover_by_tags(api, tags, horizon_hours, log, min_vol=0.0):
    """按 tag 找未来 horizon_hours 内开球的比赛市场。"""
    found = {}
    for tag in tags:
        try:
            t = api.tag_by_slug(tag)
            tag_id = t["id"]
        except Exception as e:  # noqa: BLE001
            log("discover_tag_fail", tag=tag, err=str(e)[:150])
            continue
        offset = 0
        while True:
            try:
                evs = api.events(tag_id=tag_id, closed="false", limit=100,
                                 offset=offset, order="startDate", ascending="false")
            except Exception as e:  # noqa: BLE001
                log("discover_events_fail", tag=tag, err=str(e)[:150])
                break
            if not evs:
                break
            for ev in evs:
                _absorb_event(ev, found, horizon_hours, min_vol, tag)
            if len(evs) < 100:
                break
            offset += 100
            if offset >= 500:
                break
        log("discover_tag", tag=tag, tag_id=tag_id, cum_tokens=len(found))
    return found


def discover_top_volume(api, top_n, log, min_vol=0.0):
    """休赛期兜底：直接按 24h 成交额挑最活跃的未结算市场。
    用于在没有足球赛程时也能跑通/压测整条链路。"""
    found = {}
    for page in range(3):
        try:
            mks = api.markets(closed="false", active="true", limit=100,
                              offset=page * 100, order="volume24hr", ascending="false")
        except Exception as e:  # noqa: BLE001
            log("discover_topvol_fail", err=str(e)[:150])
            break
        if not mks:
            break
        for m in mks:
            _absorb_market(m, found, None, None, min_vol, "top-volume",
                           event_slug=None, sport=None)
        if len(found) >= top_n * 2:
            break
    items = sorted(found.items(), key=lambda kv: -(kv[1]["vol24"] or 0))[:top_n]
    return dict(items)


def _absorb_event(ev, found, horizon_hours, min_vol, tag):
    sport = ev.get("sport")
    ev_slug = ev.get("slug")
    for m in ev.get("markets") or []:
        _absorb_market(m, found, horizon_hours, tag, min_vol, "tag",
                       event_slug=ev_slug, sport=sport)


def _absorb_market(m, found, horizon_hours, tag, min_vol, source,
                   event_slug=None, sport=None):
    if m.get("closed") or not m.get("active"):
        return
    if not m.get("enableOrderBook", True):
        return
    if m.get("acceptingOrders") is False:
        return
    try:
        toks = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) \
            else m.get("clobTokenIds")
        outcomes = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) \
            else (m.get("outcomes") or [])
    except Exception:  # noqa: BLE001
        return
    if not toks:
        return
    vol24 = m.get("volume24hr") or 0
    if vol24 < min_vol:
        return
    if _slug_filter and not _slug_filter.match(m.get("slug") or ""):
        return

    gst = parse_game_start(m.get("gameStartTime"))
    if horizon_hours is not None and gst is not None:
        t_minus = (gst - utcnow()).total_seconds()
        # 只要还没结束太久、且在时间窗内
        if t_minus > horizon_hours * 3600 or t_minus < -STALE_AFTER:
            return

    for i, tok in enumerate(toks):
        found[tok] = {
            "token_id": tok,
            "market_slug": m.get("slug"),
            "event_slug": event_slug,
            "question": m.get("question"),
            "outcome": outcomes[i] if i < len(outcomes) else f"idx{i}",
            "outcome_idx": i,
            "condition_id": m.get("conditionId"),
            "game_start": gst.isoformat() if gst else None,
            "tick_size": m.get("orderPriceMinTickSize"),
            "min_order_size": m.get("orderMinSize"),
            "sport": sport,
            "tag": tag,
            "source": source,
            "vol24": vol24,
            "liquidity": m.get("liquidityNum"),
            "discovered_at": utcnow().isoformat(timespec="seconds"),
        }


# ------------------------------------------------------------------ 落盘

class SnapshotWriter:
    """按 UTC 日期分文件的 NDJSON 追加写。每行一个 (时刻, token) 快照。"""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._day = None
        self._f = None
        self.written = 0

    def _rotate(self):
        day = utcnow().strftime("%Y-%m-%d")
        if day != self._day:
            if self._f:
                self._f.close()
            self._day = day
            self._f = open(os.path.join(self.root, f"depth_{day}.ndjson"),
                           "a", encoding="utf-8")

    def write(self, rec):
        self._rotate()
        self._f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.written += 1

    def flush(self):
        if self._f:
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self):
        self.flush()
        if self._f:
            self._f.close()


# ------------------------------------------------------------------ 主循环

class Collector:
    def __init__(self, args):
        self.args = args
        os.makedirs(DATA, exist_ok=True)
        self.log = Log(os.path.join(LOGS, "collector.log"))
        self.api = PolymarketRO(min_interval=args.min_interval, log=self.log)
        self.writer = SnapshotWriter(SNAP_DIR)
        self.tracked = {}
        self.state = {"next_due": {}, "fails": {}, "last_hash": {},
                      "snapshots": 0, "started": utcnow().isoformat()}
        self.stop = False
        self._load()
        signal.signal(signal.SIGINT, self._sig)
        signal.signal(signal.SIGTERM, self._sig)

    def _sig(self, *_):
        self.log("shutdown_signal")
        self.stop = True

    # ---- 断点续传 ----
    def _load(self):
        if os.path.exists(TRACK_PATH):
            try:
                self.tracked = json.load(open(TRACK_PATH, encoding="utf-8"))
                self.log("resume_tracked", n=len(self.tracked))
            except Exception as e:  # noqa: BLE001
                self.log("resume_tracked_fail", err=str(e)[:150])
        if os.path.exists(STATE_PATH):
            try:
                st = json.load(open(STATE_PATH, encoding="utf-8"))
                self.state.update(st)
                self.log("resume_state", snapshots=st.get("snapshots"),
                         due=len(st.get("next_due", {})))
            except Exception as e:  # noqa: BLE001
                self.log("resume_state_fail", err=str(e)[:150])

    def _save(self):
        atomic_write_json(STATE_PATH, self.state)
        atomic_write_json(TRACK_PATH, self.tracked)

    # ---- 发现 ----
    def refresh(self):
        a = self.args
        if a.mode == "top-volume":
            found = discover_top_volume(self.api, a.top_n, self.log, a.min_vol)
        else:
            tags = [t.strip() for t in a.tags.split(",") if t.strip()]
            found = discover_by_tags(self.api, tags, a.horizon_hours,
                                     self.log, a.min_vol)
        if self.args.max_tracked:
            found = dict(sorted(found.items(),
                                key=lambda kv: -(kv[1]["vol24"] or 0))[:self.args.max_tracked])
        added = 0
        for tok, info in found.items():
            if tok not in self.tracked:
                self.tracked[tok] = info
                self.state["next_due"][tok] = 0   # 立即采一次
                added += 1
            else:
                self.tracked[tok].update(
                    {k: info[k] for k in ("game_start", "vol24", "liquidity")
                     if info.get(k) is not None})
        # 清理：开球后超时的
        dropped = []
        for tok, info in list(self.tracked.items()):
            gst = parse_game_start(info.get("game_start"))
            if gst and (utcnow() - gst).total_seconds() > STALE_AFTER:
                dropped.append(tok)
            if self.state["fails"].get(tok, 0) >= MAX_CONSEC_FAIL:
                dropped.append(tok)
        for tok in set(dropped):
            self.tracked.pop(tok, None)
            self.state["next_due"].pop(tok, None)
            self.state["fails"].pop(tok, None)
        self.log("refresh_done", found=len(found), added=added,
                 dropped=len(set(dropped)), tracked=len(self.tracked))
        self._save()

    # ---- 一轮采集 ----
    def tick(self):
        now = time.time()
        due = [t for t in self.tracked if self.state["next_due"].get(t, 0) <= now]
        if not due:
            return 0
        got = 0
        for i in range(0, len(due), BOOKS_CHUNK):
            chunk = due[i:i + BOOKS_CHUNK]
            got += self._fetch_chunk(chunk)
        self.writer.flush()
        self.state["snapshots"] = self.state.get("snapshots", 0) + got
        self._save()
        return got

    def _fetch_chunk(self, chunk):
        capture_ms = now_ms()
        try:
            books = self.api.books(chunk)
        except Exception as e:  # noqa: BLE001
            self.log("books_fail", n=len(chunk), err=str(e)[:200])
            # 整批失败：不推进 next_due，下一轮自然重试（限流器已经退避过）
            for t in chunk:
                self.state["fails"][t] = self.state["fails"].get(t, 0) + 1
            return 0

        by_tok = {}
        for b in books or []:
            if b and b.get("asset_id"):
                by_tok[b["asset_id"]] = b
        missing = [t for t in chunk if t not in by_tok]
        # /books 会静默丢 token，单发补齐
        for t in missing[:30]:
            try:
                b = self.api.book(t)
                if b and b.get("asset_id"):
                    by_tok[b["asset_id"]] = b
            except PMError as e:
                if e.status == 404:
                    # 市场已结算 —— 深度从此不可得，停止跟踪
                    self.log("market_settled", token=t[:20],
                             slug=self.tracked.get(t, {}).get("market_slug"))
                    self.tracked.pop(t, None)
                    self.state["next_due"].pop(t, None)
                else:
                    self.state["fails"][t] = self.state["fails"].get(t, 0) + 1
            except Exception:  # noqa: BLE001
                self.state["fails"][t] = self.state["fails"].get(t, 0) + 1

        n = 0
        for tok in chunk:
            b = by_tok.get(tok)
            if b is None:
                continue
            info = self.tracked.get(tok)
            if info is None:
                continue
            self.state["fails"][tok] = 0
            snap = summarize_book(b, depth_n=self.args.depth_n)

            gst = parse_game_start(info.get("game_start"))
            t_minus = (gst - utcnow()).total_seconds() if gst else None
            snap.update({
                "capture_ms": capture_ms,
                "capture_iso": utcnow().isoformat(timespec="milliseconds"),
                "market_slug": info.get("market_slug"),
                "event_slug": info.get("event_slug"),
                "outcome": info.get("outcome"),
                "outcome_idx": info.get("outcome_idx"),
                "sport": info.get("sport"),
                "tag": info.get("tag"),
                "game_start": info.get("game_start"),
                "t_minus_sec": round(t_minus) if t_minus is not None else None,
            })
            # book_hash 没变说明挂单簿一动没动，标记出来便于压缩/去重分析
            prev = self.state["last_hash"].get(tok)
            snap["unchanged"] = (prev == snap.get("book_hash"))
            self.state["last_hash"][tok] = snap.get("book_hash")

            if not (self.args.skip_unchanged and snap["unchanged"]):
                self.writer.write(snap)
                n += 1
            interval = self.args.force_interval or cadence_for(t_minus)
            self.state["next_due"][tok] = time.time() + interval
        return n

    def run(self):
        a = self.args
        self.log("start", mode=a.mode, tags=a.tags, horizon_h=a.horizon_hours,
                 duration=a.duration, min_interval=a.min_interval,
                 snap_dir=SNAP_DIR)
        t_end = time.time() + a.duration if a.duration else None
        last_refresh = 0
        last_report = time.time()
        while not self.stop:
            if time.time() - last_refresh > a.refresh_sec:
                try:
                    self.refresh()
                except Exception as e:  # noqa: BLE001
                    self.log("refresh_fail", err=str(e)[:200])
                last_refresh = time.time()
            try:
                n = self.tick()
                if n:
                    self.log("tick", written=n, tracked=len(self.tracked),
                             total=self.state["snapshots"])
            except Exception as e:  # noqa: BLE001
                self.log("tick_fail", err=str(e)[:250])
                time.sleep(5)
            if time.time() - last_report > 300:
                self.log("heartbeat", tracked=len(self.tracked),
                         total_snapshots=self.state["snapshots"],
                         http=self.api.clob.stats)
                last_report = time.time()
            if t_end and time.time() >= t_end:
                self.log("duration_reached")
                break
            time.sleep(a.loop_sleep)
        self.writer.close()
        self._save()
        self.log("stopped", total_snapshots=self.state["snapshots"],
                 file_lines=self.writer.written, http=self.api.clob.stats)


# ------------------------------------------------------------------ 分析

def cmd_analyze(args):
    import statistics as stt
    files = sorted(os.listdir(SNAP_DIR)) if os.path.isdir(SNAP_DIR) else []
    rows = []
    for fn in files:
        if not fn.endswith(".ndjson"):
            continue
        for line in open(os.path.join(SNAP_DIR, fn), encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    print(f"读入 {len(rows)} 条快照，来自 {len(files)} 个文件")
    if not rows:
        return
    ok = [r for r in rows if r.get("spread") is not None]
    print(f"双边有报价: {len(ok)}")

    def pct(vals, q):
        vals = sorted(vals)
        if not vals:
            return None
        i = min(len(vals) - 1, int(q * (len(vals) - 1)))
        return vals[i]

    for lo, hi, name in [(0.0, 0.10, "极端 <0.10"), (0.10, 0.35, "0.10-0.35"),
                         (0.35, 0.65, "0.35-0.65 (接近五五开)"),
                         (0.65, 0.90, "0.65-0.90"), (0.90, 1.0, "极端 >0.90")]:
        sub = [r for r in ok if lo <= (r["mid"] or 0) < hi]
        if not sub:
            continue
        sp = [r["spread"] for r in sub]
        tk = [r["spread_ticks"] for r in sub]
        hs = [r["half_spread_pct"] for r in sub if r.get("half_spread_pct") is not None]
        print(f"\n[{name}] n={len(sub)}")
        print(f"  绝对价差 中位={stt.median(sp):.4f} p25={pct(sp,.25):.4f} "
              f"p75={pct(sp,.75):.4f} p90={pct(sp,.90):.4f} max={max(sp):.4f}")
        print(f"  价差(tick数) 中位={stt.median(tk):.1f} p90={pct(tk,.90):.1f}")
        if hs:
            print(f"  半价差/中价 中位={stt.median(hs):.3f}% p75={pct(hs,.75):.3f}% "
                  f"p90={pct(hs,.90):.3f}%")
    # 按市场
    per = defaultdict(list)
    for r in ok:
        per[(r.get("market_slug"), r.get("outcome"))].append(r)
    print(f"\n覆盖 {len(per)} 个 (市场,选项)")


def cmd_discover(args):
    log = Log(os.path.join(LOGS, "discover.log"))
    api = PolymarketRO(min_interval=args.min_interval, log=log)
    if args.mode == "top-volume":
        found = discover_top_volume(api, args.top_n, log, args.min_vol)
    else:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        found = discover_by_tags(api, tags, args.horizon_hours, log, args.min_vol)
    print(f"\n发现 {len(found)} 个 token:")
    seen = set()
    for tok, i in sorted(found.items(), key=lambda kv: -(kv[1]["vol24"] or 0)):
        if i["market_slug"] in seen:
            continue
        seen.add(i["market_slug"])
        print(f"  vol24={i['vol24'] or 0:>10.0f}  {str(i['market_slug'])[:60]:<60} "
              f"start={i['game_start']}")
    if args.save:
        atomic_write_json(TRACK_PATH, found)
        print("已写入", TRACK_PATH)


def main():
    p = argparse.ArgumentParser(description="Polymarket 盘口深度采集器")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--mode", default="tag", choices=["tag", "top-volume"])
        sp.add_argument("--tags", default="epl,soccer,mlb",
                        help="Gamma tag slug，逗号分隔")
        sp.add_argument("--horizon-hours", type=float, default=72,
                        help="只跟踪未来这么多小时内开球的比赛")
        sp.add_argument("--top-n", type=int, default=40)
        sp.add_argument("--min-vol", type=float, default=0.0,
                        help="24h 成交额下限")
        sp.add_argument("--min-interval", type=float, default=0.25,
                        help="全局 HTTP 最小间隔(秒)")
        sp.add_argument("--depth-n", type=int, default=10)
        sp.add_argument("--max-tracked", type=int, default=0,
                        help="跟踪 token 数上限（按成交额取前 N），0=不限")
        sp.add_argument("--force-interval", type=float, default=0,
                        help="压测用：忽略自适应节奏，固定采样间隔(秒)")
        sp.add_argument("--slug-regex", default="",
                        help="只跟踪 slug 匹配该正则的市场；填 'moneyline' 用内置的"
                             "主盘规则（滤掉 corners/exact-score/total 等玩法盘）")

    sp = sub.add_parser("run")
    common(sp)
    sp.add_argument("--duration", type=float, default=0, help="0=永久")
    sp.add_argument("--refresh-sec", type=float, default=1800)
    sp.add_argument("--loop-sleep", type=float, default=5)
    sp.add_argument("--skip-unchanged", action="store_true",
                    help="book_hash 未变则不落盘（省空间，但会丢失采样时点）")

    sp = sub.add_parser("discover")
    common(sp)
    sp.add_argument("--save", action="store_true")

    sp = sub.add_parser("analyze")

    args = p.parse_args()
    global _slug_filter
    pat = getattr(args, "slug_regex", "") or ""
    if pat == "moneyline":
        pat = MONEYLINE_RE
    if pat:
        _slug_filter = re.compile(pat)
    if args.cmd == "run":
        Collector(args).run()
    elif args.cmd == "discover":
        cmd_discover(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
