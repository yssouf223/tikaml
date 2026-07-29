"""三价之和诊断（整数精确算术，避免浮点噪声）+ 联赛水位对比"""
import json, gzip, statistics as st, collections
OUT = "/private/tmp/claude-501/-Users-yafet-Documents-github-sportalpha/18221fea-e715-4eeb-9da1-3bbc23204c32/scratchpad/pm25_all"
LG = ["epl", "laliga", "seriea", "bundesliga", "ligue1"]
NAME = {"epl": "英超", "laliga": "西甲", "seriea": "意甲", "bundesliga": "德甲", "ligue1": "法甲"}


def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(len(v) * p))]


data = {}
for lg in LG:
    rows = []
    for line in gzip.open(f"{OUT}/{lg}_series.jsonl.gz", "rt"):
        r = json.loads(line)
        mk = r["mk"]
        if not all("p" in mk[k] for k in ("home", "draw", "away")):
            continue
        s = sum(mk[k]["p"][-1] for k in ("home", "draw", "away"))  # 整数 ×10000
        rows.append({"slug": r["slug"], "vol": r["event_vol"], "sum": s,
                     "h": mk["home"]["p"][-1], "d": mk["draw"]["p"][-1], "a": mk["away"]["p"][-1],
                     "res": [mk[k]["res"] for k in ("home", "draw", "away")]})
    data[lg] = rows

print("=" * 112)
print("三价之和（赛前最后价，未归一化，整数精确）")
print(f"{'联赛':<6}{'N':>5}{'中位':>9}{'均值':>9}{'p05':>8}{'p25':>8}{'p75':>8}{'p95':>8}"
      f"{'<1':>8}{'=1':>8}{'>1':>8}{'|和-1|均值':>12}")
print("=" * 112)
diag = {}
for lg in LG:
    s = [r["sum"] for r in data[lg]]
    n = len(s)
    lt = sum(1 for x in s if x < 10000) / n
    eq = sum(1 for x in s if x == 10000) / n
    gt = sum(1 for x in s if x > 10000) / n
    mad = sum(abs(x - 10000) for x in s) / n / 10000
    diag[lg] = {"n": n, "median": st.median(s) / 1e4, "mean": sum(s) / n / 1e4,
                "pct_lt1": lt, "pct_eq1": eq, "pct_gt1": gt, "mad": mad,
                "p05": q(s, .05) / 1e4, "p95": q(s, .95) / 1e4}
    print(f"{NAME[lg]:<6}{n:>5}{st.median(s)/1e4:>9.4f}{sum(s)/n/1e4:>9.4f}"
          f"{q(s,.05)/1e4:>8.4f}{q(s,.25)/1e4:>8.4f}{q(s,.75)/1e4:>8.4f}{q(s,.95)/1e4:>8.4f}"
          f"{lt*100:>7.1f}%{eq*100:>7.1f}%{gt*100:>7.1f}%{mad*100:>11.3f}%")
print("=" * 112)
print("注：价格网格为 0.005，很多场三价和恰好 =1；用浮点求和时这批场次会被舍入噪声随机推到 <1 或 >1，")
print("    所以「<1 比例」本身不稳定。看 |和-1| 平均离差 与 =1 占比 更可靠。")

print("\n按成交量三分位看水位（|和-1| 平均离差 / 三价和中位）")
print(f"{'联赛':<6}{'低量':>22}{'中量':>22}{'高量':>22}")
for lg in LG:
    rs = sorted(data[lg], key=lambda r: r["vol"])
    n = len(rs); a, b = n // 3, 2 * n // 3
    cells = []
    for g in (rs[:a], rs[a:b], rs[b:]):
        m = sum(abs(r["sum"] - 10000) for r in g) / len(g) / 10000
        cells.append(f"{m*100:.3f}% / {st.median([r['sum'] for r in g])/1e4:.4f}")
    print(f"{NAME[lg]:<6}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

print("\n归一化后隐含概率的分布（赛前最后价 / 三价和）")
print(f"{'联赛':<6}{'主胜均值':>10}{'平局均值':>10}{'客胜均值':>10}{'实际主胜':>10}{'实际平局':>10}{'实际客胜':>10}")
for lg in LG:
    rs = data[lg]
    ph = [r["h"] / r["sum"] for r in rs]; pd = [r["d"] / r["sum"] for r in rs]; pa = [r["a"] / r["sum"] for r in rs]
    ah = sum(1 for r in rs if r["res"][0] == 1) / len(rs)
    ad = sum(1 for r in rs if r["res"][1] == 1) / len(rs)
    aa = sum(1 for r in rs if r["res"][2] == 1) / len(rs)
    print(f"{NAME[lg]:<6}{sum(ph)/len(ph)*100:>9.1f}%{sum(pd)/len(pd)*100:>9.1f}%{sum(pa)/len(pa)*100:>9.1f}%"
          f"{ah*100:>9.1f}%{ad*100:>9.1f}%{aa*100:>9.1f}%")

json.dump(diag, open(f"{OUT}/diagnostics.json", "w"), ensure_ascii=False, indent=1)
print("\n→ 写入", f"{OUT}/diagnostics.json")
