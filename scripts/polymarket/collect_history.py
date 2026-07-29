"""任务B：抓 Polymarket 五大联赛 25/26 单场 1x2 的完整分钟级价格序列

存放：scratchpad/pm25_all/
  {lg}_series.jsonl.gz  每行一场，含三个市场的完整时间序列（紧凑编码）
  {lg}_summary.json     每场一条：赛前最后价 + 交易量 + 结果 + 序列统计
"""
import json, re, gzip, time, os, sys, threading, urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

G = "https://gamma-api.polymarket.com"
C = "https://clob.polymarket.com"
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "Accept": "application/json"}
OUT = "/private/tmp/claude-501/-Users-yafet-Documents-github-sportalpha/18221fea-e715-4eeb-9da1-3bbc23204c32/scratchpad/pm25_all"
os.makedirs(OUT, exist_ok=True)

# 联赛 → (tag_id, slug 前缀)
LEAGUES = {
    "epl": (306, "epl"),
    "laliga": (780, "lal"),
    "seriea": (101962, "sea"),
    "bundesliga": (1494, "bun"),
    "ligue1": (102070, "fl1"),
}
SEASON_MIN, SEASON_MAX = "2025-08-01", "2026-07-01"

_lk = threading.Lock()
_last = [0.0]
MIN_GAP = float(os.environ.get("PM_GAP", "0.42"))  # 进程内最小请求间隔（秒），两进程并行 → 全局约 0.21s


def _throttle():
    with _lk:
        now = time.time()
        w = _last[0] + MIN_GAP - now
        if w > 0:
            time.sleep(w)
            now = time.time() + 0  # noqa
        _last[0] = time.time()


def get(u, tries=3):
    for i in range(tries):
        _throttle()
        try:
            r = urllib.request.Request(u, headers=H)
            with urllib.request.urlopen(r, timeout=90) as x:
                return json.load(x)
        except Exception as e:
            if i == tries - 1:
                return {"__err": str(e)[:120]}
            time.sleep(1.5 * (i + 1))


def fetch_events(tag_id):
    evs, seen = [], set()
    for off in range(0, 4000, 100):
        d = get(f"{G}/events?tag_id={tag_id}&closed=true&limit=100&offset={off}"
                f"&end_date_min={SEASON_MIN}&end_date_max={SEASON_MAX}")
        if not isinstance(d, list) or not d:
            break
        for e in d:
            if e.get("slug") not in seen:
                seen.add(e["slug"])
                evs.append(e)
        if len(d) < 100:
            break
    return evs


RE_WIN = re.compile(r"^Will (.+?) win on \d{4}-\d{2}-\d{2}\?$")


def parse_event(e, pre):
    """返回 (home, away, [ (key, market) x3 ]) 或 None"""
    if not re.match(rf"^{pre}-[a-z0-9]{{2,5}}-[a-z0-9]{{2,5}}-\d{{4}}-\d{{2}}-\d{{2}}$", e.get("slug", "")):
        return None
    mks = e.get("markets") or []
    if len(mks) != 3:
        return None
    wins, draw = [], None
    for m in mks:
        q = (m.get("question") or "").strip()
        if "end in a draw" in q.lower():
            draw = m
        else:
            mm = RE_WIN.match(q)
            if not mm:
                return None
            wins.append((mm.group(1).strip(), m))
    if draw is None or len(wins) != 2:
        return None
    title = e.get("title") or ""
    home = wins[0][0]
    if " vs. " in title:
        home = title.split(" vs. ")[0].strip()
    if wins[0][0] == home:
        (hn, hm), (an, am) = wins[0], wins[1]
    else:
        (an, am), (hn, hm) = wins[0], wins[1]
    return hn, an, [("home", hn, hm), ("draw", None, draw), ("away", an, am)]


def toks_of(m):
    t = m.get("clobTokenIds")
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except Exception:
            return None
    return t if t else None


def fetch_series(tok, kick_ts):
    """14 天窗口、fidelity=1，返回 (t0, dts, ps_int) 紧凑序列"""
    st = kick_ts - 14 * 86400 + 3600  # 略小于 14d 上限
    h = get(f"{C}/prices-history?market={tok}&startTs={st}&endTs={kick_ts}&fidelity=1")
    pts = (h or {}).get("history") or []
    if not pts:
        return None
    ts = [int(p["t"]) for p in pts]
    ps = [int(round(float(p["p"]) * 10000)) for p in pts]
    dts = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
    return ts[0], dts, ps


def settled(m):
    try:
        op = m.get("outcomePrices")
        if isinstance(op, str):
            op = json.loads(op)
        return float(op[0]) if op else None
    except Exception:
        return None


def run_league(lg):
    tag_id, pre = LEAGUES[lg]
    evs = fetch_events(tag_id)
    games = []
    for e in evs:
        p = parse_event(e, pre)
        if p:
            games.append((e, p))
    games.sort(key=lambda x: x[0].get("endDate") or "")
    print(f"[{lg}] 事件 {len(evs)} → 单场1x2 {len(games)}", flush=True)
    if not games:
        return

    fp = gzip.open(f"{OUT}/{lg}_series.jsonl.gz", "wt", compresslevel=6)
    wlock = threading.Lock()
    summ, stats = [], {"ok": 0, "miss": 0, "done": 0}

    def work(item):
        e, (hn, an, trio) = item
        kick = e["endDate"]
        kts = int(datetime.fromisoformat(kick.replace("Z", "+00:00")).timestamp())
        rec = {"slug": e["slug"], "league": lg, "title": e.get("title"),
               "home": hn, "away": an, "kick": kick, "kick_ts": kts,
               "event_vol": float(e.get("volume") or 0), "mk": {}}
        srow = {"slug": e["slug"], "league": lg, "title": e.get("title"),
                "home": hn, "away": an, "kick": kick[:19], "vol": rec["event_vol"]}
        for key, name, m in trio:
            toks = toks_of(m)
            mv = float(m.get("volume") or 0)
            res = settled(m)
            if not toks:
                stats["miss"] += 1
                rec["mk"][key] = {"err": "no_token", "vol": mv, "res": res}
                srow[key] = None
                srow[key + "|n"] = 0
                continue
            s = fetch_series(toks[0], kts)
            if s is None:
                stats["miss"] += 1
                rec["mk"][key] = {"err": "no_hist", "tok": toks[0], "vol": mv, "res": res}
                srow[key] = None
                srow[key + "|n"] = 0
                continue
            t0, dts, ps = s
            stats["ok"] += 1
            rec["mk"][key] = {"tok": toks[0], "team": name, "q": m.get("question"),
                              "vol": mv, "res": res, "t0": t0, "dt": dts, "p": ps}
            srow[key] = ps[-1] / 10000.0
            srow[key + "|n"] = len(ps)
            srow[key + "|vol"] = mv
            srow[key + "|res"] = res
            srow[key + "|t0"] = t0
            srow[key + "|last_t"] = t0 + sum(dts)
        with wlock:
            fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            summ.append(srow)
            stats["done"] += 1
            if stats["done"] % 25 == 0:
                print(f"  [{lg}] {stats['done']}/{len(games)} ok={stats['ok']} miss={stats['miss']}", flush=True)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(work, games))
    fp.close()
    summ.sort(key=lambda r: r["kick"])
    json.dump(summ, open(f"{OUT}/{lg}_summary.json", "w"), ensure_ascii=False)
    sz = os.path.getsize(f"{OUT}/{lg}_series.jsonl.gz") / 1e6
    print(f"[{lg}] 完成 {len(summ)} 场，市场 ok={stats['ok']} miss={stats['miss']}，"
          f"gz={sz:.1f}MB", flush=True)


if __name__ == "__main__":
    for lg in (sys.argv[1:] or list(LEAGUES)):
        t = time.time()
        run_league(lg)
        print(f"[{lg}] 用时 {time.time()-t:.0f}s\n", flush=True)
