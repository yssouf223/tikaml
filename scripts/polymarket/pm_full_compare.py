"""Polymarket vs Pinnacle vs 模型 —— 25/26 五大联赛全量三方对比。

背景（交接文档 §6 第 1 步）
--------------------------
上一轮只在英超 209 场上粗测过：PM 赛前最后价 RPS 0.19580 vs Pinnacle 收盘
0.19971，配对差 −0.00391 ± 0.00210（1.86 SE，差一点点显著）。
本脚本用 25/26 全量约 1750 场重做，回答：那个效应还在吗？显著了吗？

数据源与对齐
------------
1. Polymarket: data/polymarket/summary/*.json（赛前最后价 + 结算结果）。
   去掉改期造成的重复记录（同联赛同主客队保留序列点数多的），去掉 3 场升降级附加赛。
2. Pinnacle: data/odds/{E0,SP1,I1,D1,F1}_2526.csv 的 PS*（赛前）与 PSC*（收盘）。
3. 模型: features_v3.csv 上按 harness 协议 forward-chain（train < 2025-2026，
   验证 2024-2025），LGBMPoissonModel(rho=-0.108, temperature=0.90)，多种子。

三个数据源队名各不相同，按联赛做归一化 + 相似度的 1:1 双射映射（断言无冲突），
join 键 = (league, home, away)——常规赛内每季唯一，不依赖日期。
对齐质量由赛果交叉验证兜底：PM 结算结果 / football-data FTR / Opta 比分
三方必须一致，任何不一致都会打印出来。

去水口径：三家都用比例归一（与上一轮及 build_features.py 一致）。

用法
----
    .venv/bin/python scripts/polymarket/pm_full_compare.py --seeds=5
"""

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, LGB_PARAMS  # noqa: E402
from src.evaluation import match_outcome  # noqa: E402

PM_DIR = ROOT / "data/polymarket/summary"
ODDS_DIR = ROOT / "data/odds"
FEATURES = ROOT / "data/opta/processed/features_v3.csv"
OUT_CSV = ROOT / "data/polymarket/full_compare_2526.csv"
OUT_TXT = ROOT / "scripts/polymarket/reports/full_compare_2526.txt"

# 联赛三方代号：PM 文件名 / football-data 文件前缀 / features 里的 league
LEAGUES = [
    ("epl", "E0", "EPL", "英超"),
    ("laliga", "SP1", "LL", "西甲"),
    ("seriea", "I1", "SEA", "意甲"),
    ("bundesliga", "D1", "BUN", "德甲"),
    ("ligue1", "F1", "LI1", "法甲"),
]

# 升降级附加赛（README_format.md），不属于常规赛
PLAYOFF_SLUGS = {
    "bun-pad-wol-2026-05-25",
    "fl1-se-nic-2026-05-26",
    "fl1-nic-se-2026-05-29",
}

# 归一化后仍难自动配对的已知别名（PM 名 / football-data 名 / Opta 名 → 统一形）。
# 三方队名全量核对过（PM title 变体 / FD 短名 / Opta 名），城市词重叠的
# （Rayo/Atlético、Espanyol/Barcelona、Inter/Milan）必须显式钉死，不能靠相似度。
ALIASES = {
    # 英超
    "wolverhampton wanderers": "wolves", "leeds united": "leeds",
    "newcastle united": "newcastle", "tottenham hotspur": "tottenham",
    "west ham united": "west ham", "nott m forest": "nottingham forest",
    "manchester city": "man city", "manchester united": "man united",
    # 西甲
    "athletic club": "athletic bilbao", "ath bilbao": "athletic bilbao",
    "atletico de madrid": "atletico madrid", "ath madrid": "atletico madrid",
    "rayo vallecano": "vallecano", "rayo vallecano madrid": "vallecano",
    "real betis": "betis", "espanyol": "espanol",
    "espanyol barcelona": "espanol", "celta vigo": "celta",
    "real sociedad": "sociedad", "real oviedo": "oviedo",
    "deportivo alaves": "alaves",
    # 意甲
    "internazionale": "inter", "internazionale milano": "inter",
    "hellas verona": "verona",
    # 德甲
    "bayern munchen": "bayern munich", "borussia dortmund": "dortmund",
    "eintracht frankfurt": "ein frankfurt", "bayer leverkusen": "leverkusen",
    "borussia monchengladbach": "m gladbach", "borussia m gladbach": "m gladbach",
    "hamburger": "hamburg",
    # 法甲
    "olympique lyonnais": "lyon", "olympique marseille": "marseille",
    "paris saint germain": "paris sg", "racing lens": "lens",
    "rennais": "rennes", "brestois": "brest",
    "strasbourg alsace": "strasbourg", "saint etienne": "st etienne",
}

_STOP = {"fc", "cf", "afc", "ac", "as", "ssc", "us", "ss", "sc", "rc", "rcd",
         "cd", "ud", "sd", "ca", "club", "de", "sco", "asc", "ol", "losc",
         "osc", "ogc", "aj", "stade", "hove", "albion", "calcio", "bc", "cfc",
         "acf", "balompie", "futbol", "fsv", "tsg", "sv", "vfb", "vfl", "bv"}


def norm_name(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _STOP and not t.isdigit()]
    out = " ".join(toks)
    return ALIASES.get(out, out)


def build_bijection(src_names, dst_names, tag):
    """src → dst 的 1:1 映射（在归一化名层面），返回 raw src 名 → raw dst 名。

    PM 的 title 里同一队有多个写法（带不带 FC 等），先归一化去重再配对。
    """
    src_norm = {}
    for a in src_names:
        src_norm.setdefault(norm_name(a), []).append(a)
    dst_norm = {norm_name(b): b for b in dst_names}
    if len(dst_norm) != len(dst_names):
        raise RuntimeError(f"{tag}: dst 归一化后重名")
    pairs = [(SequenceMatcher(None, na, nb).ratio() if na != nb else 2.0, na, nb)
             for na in src_norm for nb in dst_norm]
    pairs.sort(reverse=True)
    # 第一遍：1:1 贪心，让每个 dst 队名拿到最像的 src 归一名
    used_a, used_b, mapping = set(), set(), {}
    for score, na, nb in pairs:
        if na in used_a or nb in used_b:
            continue
        mapping[na] = (nb, score)
        used_a.add(na)
        used_b.add(nb)
    # 第二遍：剩余 src 变体（同队的另一种写法）自由挂靠最佳 dst
    for na in set(src_norm) - used_a:
        best = max(((SequenceMatcher(None, na, nb).ratio(), nb) for nb in dst_norm))
        mapping[na] = (best[1], best[0])
    low = {a: v for a, v in mapping.items() if v[1] < 0.6}
    if low:
        print(f"  [warn] {tag} 低置信映射: "
              + "; ".join(f"{a}→{v[0]}({v[1]:.2f})" for a, v in low.items()))
    return {raw: dst_norm[mapping[na][0]]
            for na, raws in src_norm.items() for raw in raws}


def load_pm():
    rows = []
    for pm_lg, _, feat_lg, _ in LEAGUES:
        for r in json.load(open(PM_DIR / f"{pm_lg}_summary.json")):
            if r["slug"] in PLAYOFF_SLUGS:
                continue
            res = [r["home|res"], r["draw|res"], r["away|res"]]
            if sorted(res) != [0.0, 0.0, 1.0]:
                raise RuntimeError(f"{r['slug']}: 结算结果异常 {res}")
            title = r["title"]
            home, away = re.split(r"\s+vs\.?\s+", title, maxsplit=1)
            p = np.array([r["home"], r["draw"], r["away"]], dtype=float)
            rows.append({
                "slug": r["slug"], "league": feat_lg,
                "pm_home": home.strip(), "pm_away": away.strip(),
                "slug_date": r["slug"][-10:], "n_points": r["home|n"],
                "vol": r["vol"], "pm_sum": p.sum(),
                "pm_h": p[0] / p.sum(), "pm_d": p[1] / p.sum(),
                "pm_a": p[2] / p.sum(),
                "pm_outcome": int(np.argmax(res)),
            })
    df = pd.DataFrame(rows)
    # 改期重复：同 (league, home, away) 保留序列点数多的那条
    df = (df.sort_values("n_points", ascending=False)
            .drop_duplicates(["league", "pm_home", "pm_away"])
            .reset_index(drop=True))
    return df


def load_fd():
    rows = []
    for _, fd_lg, feat_lg, _ in LEAGUES:
        odf = pd.read_csv(ODDS_DIR / f"{fd_lg}_2526.csv")
        odf["date"] = pd.to_datetime(odf["Date"], format="%d/%m/%Y")
        for r in odf.itertuples():
            rows.append({
                "league": feat_lg, "fd_home": r.HomeTeam, "fd_away": r.AwayTeam,
                "fd_date": r.date, "fd_outcome": {"H": 0, "D": 1, "A": 2}[r.FTR],
                "ps_h": r.PSH, "ps_d": r.PSD, "ps_a": r.PSA,
                "psc_h": r.PSCH, "psc_d": r.PSCD, "psc_a": r.PSCA,
                "b365c_h": r.B365CH, "b365c_d": r.B365CD, "b365c_a": r.B365CA,
                "avgc_h": r.AvgCH, "avgc_d": r.AvgCD, "avgc_a": r.AvgCA,
                "maxc_h": r.MaxCH, "maxc_d": r.MaxCD, "maxc_a": r.MaxCA,
                "avg_h": r.AvgH, "avg_d": r.AvgD, "avg_a": r.AvgA,
                "b365_h": r.B365H, "b365_d": r.B365D, "b365_a": r.B365A,
            })
    return pd.DataFrame(rows)


def devig(df, prefix):
    inv = np.column_stack([1.0 / df[f"{prefix}_{k}"] for k in "hda"])
    return inv / inv.sum(axis=1, keepdims=True)


def rps_per_match(probs, outcomes):
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    cum_a = np.cumsum(onehot, axis=1)[:, :2]
    return ((cum_p - cum_a) ** 2).sum(axis=1) / 2.0


def train_and_predict(feat, seeds):
    """harness 协议 forward-chain，test=2025-2026，返回 {match_id: mean_probs}。"""
    train_all = feat[(feat["season"] >= "2016-2017") & (feat["season"] < "2025-2026")]
    prev = sorted(train_all["season"].unique())[-1]
    val = train_all[train_all["season"] == prev]
    train = train_all[train_all["season"] < prev]
    test = feat[feat["season"] == "2025-2026"]
    print(f"  训练 {len(train)} | 验证({prev}) {len(val)} | 测试(25/26) {len(test)}")

    all_probs = []
    for s in range(seeds):
        params = LGB_PARAMS.copy()
        params.update(random_state=s, bagging_seed=s,
                      feature_fraction_seed=s, data_random_seed=s)
        model = LGBMPoissonModel(rho=-0.108, temperature=0.90,
                                 params=params, feature_list=list(FEATURE_COLS))
        model.fit(train, val_df=val)
        all_probs.append(model.predict_1x2(test))
        print(f"  seed {s} 完成")
    return test["match_id"].values, np.stack(all_probs)  # (seeds, n, 3)


def paired(tag, a, b, out):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    z = d.mean() / se if se > 0 else 0.0
    verdict = "显著" if abs(z) > 2 else "无法区分"
    out.append(f"  {tag:<28} Δ {d.mean():+.5f} ± {se:.5f}  z={z:+.2f}  {verdict}")
    return d.mean(), se, z


def main():
    seeds = 5
    for a in sys.argv[1:]:
        if a.startswith("--seeds="):
            seeds = int(a.split("=", 1)[1])

    out = ["=" * 78,
           "Polymarket vs Pinnacle vs 模型 — 25/26 五大联赛全量三方对比",
           "=" * 78]

    pm, fd = load_pm(), load_fd()
    feat = pd.read_csv(FEATURES, low_memory=False)
    f26 = feat[feat["season"] == "2025-2026"][
        ["match_id", "league", "date", "home_team", "away_team",
         "home_goals", "away_goals"]].copy()
    out.append(f"\nPM 常规赛记录 {len(pm)} | football-data {len(fd)} | features 25/26 {len(f26)}")

    # ── 按联赛建队名双射并 join ──────────────────────────────────────────
    merged = []
    for _, _, feat_lg, cn in LEAGUES:
        p = pm[pm["league"] == feat_lg].copy()
        d = fd[fd["league"] == feat_lg].copy()
        f = f26[f26["league"] == feat_lg].copy()
        pm_teams = sorted(set(p["pm_home"]) | set(p["pm_away"]))
        fd_teams = sorted(set(d["fd_home"]) | set(d["fd_away"]))
        ft_teams = sorted(set(f["home_team"]) | set(f["away_team"]))
        pm2fd = build_bijection(pm_teams, fd_teams, f"{cn} PM→FD")
        ft2fd = build_bijection(ft_teams, fd_teams, f"{cn} FEAT→FD")
        p["k_home"] = p["pm_home"].map(pm2fd)
        p["k_away"] = p["pm_away"].map(pm2fd)
        f["k_home"] = f["home_team"].map(ft2fd)
        f["k_away"] = f["away_team"].map(ft2fd)
        m = p.merge(d, left_on=["league", "k_home", "k_away"],
                    right_on=["league", "fd_home", "fd_away"], how="inner")
        m = m.merge(f, on=["league", "k_home", "k_away"], how="inner")
        out.append(f"  {cn}: PM {len(p)} × FD {len(d)} × FEAT {len(f)} → 对齐 {len(m)}")
        merged.append(m)
    m = pd.concat(merged, ignore_index=True)

    # ── 赛果三方交叉验证（对齐质量的硬检验）────────────────────────────────
    m["ft_outcome"] = [match_outcome(h, a) for h, a in
                       zip(m["home_goals"], m["away_goals"])]
    bad = m[(m["pm_outcome"] != m["fd_outcome"]) | (m["pm_outcome"] != m["ft_outcome"])]
    if len(bad):
        out.append(f"\n[FAIL] {len(bad)} 场赛果不一致（对齐错误或数据问题）：")
        for r in bad.itertuples():
            out.append(f"  {r.slug} PM={r.pm_outcome} FD={r.fd_outcome} FEAT={r.ft_outcome}"
                       f" ({r.pm_home} vs {r.pm_away})")
        print("\n".join(out))
        sys.exit(1)   # 错配场次有 1/3 概率碰对赛果混进样本，必须修到 0 再出报告
    out.append(f"\n赛果交叉验证：{len(m)} 场 PM/FD/Opta 三方全部一致 ✓")
    m = m.reset_index(drop=True)

    # 两个子集：全季（Avg/Max/B365 收盘 100% 覆盖）；Pinnacle 子集
    # （football-data 的 Pinnacle 五联赛都在 2026-01-08 后整段消失，非采集问题）
    pin_cols = [f"{p}_{k}" for p in ("ps", "psc") for k in "hda"]
    alt_cols = [f"{p}_{k}" for p in ("b365c", "avgc", "maxc") for k in "hda"]
    n_alt_miss = int(m[alt_cols].isna().any(axis=1).sum())
    if n_alt_miss:
        out.append(f"[warn] 全季子集仍有 {n_alt_miss} 场缺 B365C/AvgC/MaxC，已剔除")
        m = m.dropna(subset=alt_cols).reset_index(drop=True)
    has_pin = m[pin_cols].notna().all(axis=1).values
    out.append(f"全季对齐 {len(m)} 场 | 其中含 Pinnacle 的 {has_pin.sum()} 场"
               f"（Pinnacle 缺口集中在 2026-01-08 之后）")

    # ── 模型预测 ─────────────────────────────────────────────────────────
    print("\n训练模型（forward-chain, test=2025-2026）...")
    mids, probs_seeds = train_and_predict(feat, seeds)
    idx = {mid: i for i, mid in enumerate(mids)}
    rows_idx = np.array([idx[x] for x in m["match_id"]])
    model_probs = probs_seeds[:, rows_idx, :]          # (seeds, n, 3)
    mp_mean = model_probs.mean(axis=0)

    y = m["pm_outcome"].values
    lg_names = dict((l[2], l[3]) for l in LEAGUES)
    mk = f"模型(seed 均值×{seeds})"

    def block(mask, title, sources, key_pair):
        """一个子集的完整报告：各源 RPS、按联赛 macro、配对比较。"""
        idx_ = np.where(mask)[0]
        yy = y[idx_]
        R = {k: rps_per_match(p[idx_], yy) for k, p in sources.items()}
        out.append(f"\n{'─' * 78}\n{title}（N={len(idx_)}）\n{'─' * 78}")
        macro = {k: [] for k in R}
        for lg in [l[2] for l in LEAGUES]:
            lmask = (m["league"].values == lg) & mask
            for k in R:
                macro[k].append(rps_per_match(sources[k][lmask], y[lmask]).mean())
        for k, v in R.items():
            out.append(f"  {k:<28} micro {v.mean():.5f} | macro {np.mean(macro[k]):.5f}")
        out.append("")
        for a, b in key_pair:
            paired(f"{a} − {b}", R[a], R[b], out)
        return R, idx_

    # 子集 1：Pinnacle 可用（严格复现上一轮 209 场口径，样本 ×4.5）
    src_pin = {"PM 赛前最后价": m[["pm_h", "pm_d", "pm_a"]].values,
               "Pinnacle 收盘(PSC)": devig(m, "psc"),
               "Pinnacle 赛前(PS)": devig(m, "ps"),
               "市场均值收盘(AvgC)": devig(m, "avgc"),
               mk: mp_mean}
    R1, idx1 = block(has_pin, "子集 A：Pinnacle 可用（≈前半季）", src_pin,
                     [("PM 赛前最后价", "Pinnacle 收盘(PSC)"),
                      ("PM 赛前最后价", "Pinnacle 赛前(PS)"),
                      ("PM 赛前最后价", "市场均值收盘(AvgC)"),
                      ("Pinnacle 收盘(PSC)", "Pinnacle 赛前(PS)"),
                      (mk, "Pinnacle 收盘(PSC)"), (mk, "PM 赛前最后价")])

    out.append("\n  PM − Pinnacle收盘，按联赛：")
    for lg in [l[2] for l in LEAGUES]:
        lmask = (m["league"].values == lg) & has_pin
        paired(f"  {lg_names[lg]} (N={lmask.sum()})",
               rps_per_match(src_pin["PM 赛前最后价"][lmask], y[lmask]),
               rps_per_match(src_pin["Pinnacle 收盘(PSC)"][lmask], y[lmask]), out)

    out.append(f"\n  参考：上一轮英超 209 场 PM−PSC = −0.00391 ± 0.00210 (z=−1.86)")

    # 子集 2：全季（无 Pinnacle，用 Avg/Max/B365 收盘）
    src_all = {"PM 赛前最后价": m[["pm_h", "pm_d", "pm_a"]].values,
               "市场均值收盘(AvgC)": devig(m, "avgc"),
               "最优价收盘(MaxC)": devig(m, "maxc"),
               "B365 收盘(B365C)": devig(m, "b365c"),
               mk: mp_mean}
    R2, _ = block(np.ones(len(m), bool), "子集 B：全季", src_all,
                  [("PM 赛前最后价", "市场均值收盘(AvgC)"),
                   ("PM 赛前最后价", "最优价收盘(MaxC)"),
                   ("PM 赛前最后价", "B365 收盘(B365C)"),
                   (mk, "市场均值收盘(AvgC)"), (mk, "PM 赛前最后价")])

    out.append("\n  PM − 市场均值收盘(AvgC)，按联赛：")
    for lg in [l[2] for l in LEAGUES]:
        lmask = (m["league"].values == lg)
        paired(f"  {lg_names[lg]} (N={lmask.sum()})",
               rps_per_match(src_all["PM 赛前最后价"][lmask], y[lmask]),
               rps_per_match(src_all["市场均值收盘(AvgC)"][lmask], y[lmask]), out)

    # 模型配对的跨 seed 稳定性（全季 vs AvgC）
    model_rps_seeds = np.stack([rps_per_match(model_probs[s], y)
                                for s in range(seeds)])
    out.append(f"\n  模型(逐 seed) 全季 micro: "
               + ", ".join(f"{model_rps_seeds[s].mean():.5f}" for s in range(seeds))
               + f"  (跨 seed σ={model_rps_seeds.mean(axis=1).std(ddof=1):.5f})")
    out.append("  模型 − AvgC 收盘，逐 seed：")
    avgc_rps = rps_per_match(devig(m, "avgc"), y)
    for s in range(seeds):
        paired(f"  seed {s}", model_rps_seeds[s], avgc_rps, out)

    # ── 落盘 ─────────────────────────────────────────────────────────────
    save = m[["slug", "match_id", "league", "date", "pm_home", "pm_away",
              "vol", "pm_sum", "pm_outcome",
              "pm_h", "pm_d", "pm_a",
              "ps_h", "ps_d", "ps_a", "psc_h", "psc_d", "psc_a",
              "b365c_h", "b365c_d", "b365c_a",
              "avgc_h", "avgc_d", "avgc_a", "maxc_h", "maxc_d", "maxc_a",
              "avg_h", "avg_d", "avg_a", "b365_h", "b365_d", "b365_a"]].copy()
    for i, k in enumerate("hda"):
        save[f"model_prob_{k}"] = mp_mean[:, i]
    save["rps_pm"] = R2["PM 赛前最后价"]
    save["rps_avgc"] = R2["市场均值收盘(AvgC)"]
    save["rps_model"] = R2[mk]
    save["has_pinnacle"] = has_pin
    save.to_csv(OUT_CSV, index=False)
    out.append(f"\n逐场明细已写入 {OUT_CSV}")

    report = "\n".join(out)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(report + "\n")
    print(report)
    print(f"\n报告已写入 {OUT_TXT}")


if __name__ == "__main__":
    main()
