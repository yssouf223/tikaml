"""任务B 汇总报告：覆盖率 + 三价之和诊断"""
import json, gzip, os, statistics as st, collections
OUT = "/private/tmp/claude-501/-Users-yafet-Documents-github-sportalpha/18221fea-e715-4eeb-9da1-3bbc23204c32/scratchpad/pm25_all"
LG = ["epl", "laliga", "seriea", "bundesliga", "ligue1"]
NAME = {"epl": "英超", "laliga": "西甲", "seriea": "意甲", "bundesliga": "德甲", "ligue1": "法甲"}


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else float("nan")


print("=" * 108)
print(f"{'联赛':<6}{'场次':>5}{'三价齐全':>9}{'日期范围':>26}{'成交量中位':>12}{'量p10':>9}{'量p90':>9}{'序列点数中位':>13}")
print("=" * 108)
tot = collections.Counter()
rows = {}
for lg in LG:
    f = f"{OUT}/{lg}_summary.json"
    if not os.path.exists(f):
        print(f"{NAME[lg]:<6} —— 缺文件")
        continue
    d = json.load(open(f))
    rows[lg] = d
    full = [r for r in d if all(r.get(k) is not None for k in ("home", "draw", "away"))]
    ds = sorted(r["kick"][:10] for r in d)
    vols = [r["vol"] for r in d]
    npts = [r.get("home|n", 0) + r.get("draw|n", 0) + r.get("away|n", 0) for r in full]
    tot["g"] += len(d); tot["f"] += len(full)
    print(f"{NAME[lg]:<6}{len(d):>5}{len(full):>9}{ds[0]+' → '+ds[-1]:>28}"
          f"{st.median(vols)/1e4:>10.1f}万{q(vols,0.1)/1e4:>8.1f}万{q(vols,0.9)/1e4:>8.1f}万"
          f"{int(st.median(npts))//3 if npts else 0:>13}")
print("=" * 108)
print(f"合计 {tot['g']} 场，三价齐全 {tot['f']} 场")

print("\n" + "=" * 96)
print("三价之和（未归一化）诊断 —— 赛前最后价")
print(f"{'联赛':<6}{'N':>5}{'中位':>9}{'均值':>9}{'p05':>8}{'p25':>8}{'p75':>8}{'p95':>8}{'<1 比例':>10}{'<0.98':>9}{'>1.02':>9}")
print("=" * 96)
diag = {}
for lg in LG:
    d = rows.get(lg)
    if not d:
        continue
    s = [r["home"] + r["draw"] + r["away"] for r in d
         if all(r.get(k) is not None for k in ("home", "draw", "away"))]
    if not s:
        continue
    lt1 = sum(1 for x in s if x < 1) / len(s)
    diag[lg] = {"n": len(s), "median": st.median(s), "mean": sum(s) / len(s),
                "pct_lt1": lt1, "p05": q(s, .05), "p95": q(s, .95)}
    print(f"{NAME[lg]:<6}{len(s):>5}{st.median(s):>9.4f}{sum(s)/len(s):>9.4f}"
          f"{q(s,.05):>8.4f}{q(s,.25):>8.4f}{q(s,.75):>8.4f}{q(s,.95):>8.4f}"
          f"{lt1*100:>9.1f}%{sum(1 for x in s if x<0.98)/len(s)*100:>8.1f}%"
          f"{sum(1 for x in s if x>1.02)/len(s)*100:>8.1f}%")
print("=" * 96)

# 按成交量分组看水位
print("\n三价和 vs 成交量分层（中位数 / <1 比例）")
print(f"{'联赛':<6}{'低量(下1/3)':>22}{'中量':>22}{'高量(上1/3)':>22}")
for lg in LG:
    d = rows.get(lg)
    if not d:
        continue
    rs = [(r["vol"], r["home"] + r["draw"] + r["away"]) for r in d
          if all(r.get(k) is not None for k in ("home", "draw", "away"))]
    rs.sort()
    n = len(rs); a, b = n // 3, 2 * n // 3
    cells = []
    for grp in (rs[:a], rs[a:b], rs[b:]):
        v = [x[1] for x in grp]
        cells.append(f"{st.median(v):.4f} / {sum(1 for x in v if x<1)/len(v)*100:.0f}%")
    print(f"{NAME[lg]:<6}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

# 序列覆盖：赛前最后一个点距开球多久
print("\n序列时效（最后一个价格点距开球，分钟）与序列长度")
print(f"{'联赛':<6}{'滞后中位':>10}{'滞后p90':>10}{'序列时长中位(小时)':>20}{'点数中位':>10}")
for lg in LG:
    d = rows.get(lg)
    if not d:
        continue
    lags, spans, npt = [], [], []
    for r in d:
        for k in ("home", "draw", "away"):
            if r.get(k) is None:
                continue
            import datetime as dt
            kt = int(dt.datetime.fromisoformat(r["kick"]).replace(tzinfo=dt.timezone.utc).timestamp())
            lags.append((kt - r[k + "|last_t"]) / 60)
            spans.append((r[k + "|last_t"] - r[k + "|t0"]) / 3600)
            npt.append(r[k + "|n"])
    print(f"{NAME[lg]:<6}{st.median(lags):>10.1f}{q(lags,.9):>10.1f}{st.median(spans):>20.1f}{int(st.median(npt)):>10}")

# 场次异常：重复对阵（联赛应恰好每组主客一次）
print("\n重复/多余对阵（同一主客组合出现多次 → 改期或重复事件）：")
for lg in LG:
    d = rows.get(lg)
    if not d:
        continue
    c = collections.Counter((r["home"], r["away"]) for r in d)
    dup = {k: v for k, v in c.items() if v > 1}
    exp = {"epl": 380, "laliga": 380, "seriea": 380, "bundesliga": 306, "ligue1": 306}[lg]
    print(f"  {NAME[lg]}: {len(d)} 场 (应 {exp})，重复对阵 {len(dup)}")
    for k in dup:
        for r in d:
            if (r["home"], r["away"]) == k:
                print(f"      {r['slug']:<30} {r['title']} vol={r['vol']:,.0f}")

# 结算结果一致性（三个市场应恰好一个 res=1）
print("\n结算一致性（三个市场恰好一个 res=1）：")
for lg in LG:
    d = rows.get(lg)
    if not d:
        continue
    bad = 0
    for r in d:
        rs = [r.get(k + "|res") for k in ("home", "draw", "away")]
        if sum(1 for x in rs if x == 1.0) != 1:
            bad += 1
    print(f"  {NAME[lg]}: 异常 {bad} / {len(d)}")

json.dump(diag, open(f"{OUT}/diagnostics.json", "w"), ensure_ascii=False, indent=1)

# 文件体量
print("\n文件：")
for f in sorted(os.listdir(OUT)):
    print(f"  {f:<32}{os.path.getsize(OUT+'/'+f)/1e6:>10.1f} MB")
