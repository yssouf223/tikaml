"""实验框架：在统一的 forward-chain 协议下比较模型变体，输出带标准误的 ΔRPS。

为什么需要它
------------
交接文档 §7.2 记录了这个项目最容易犯的错：把噪声当结论。历史上「UCL 合并训练更差
+0.0002/+0.0004」被当成定论，实际就在噪声范围内。要避免重蹈覆辙，每个实验都必须：

  1. 跟基线用**完全相同的测试场次**（否则不可比）
  2. 做**配对比较**——逐场 RPS 相减再求标准误，比两个独立均值相减的 SE 小得多
  3. 先知道**噪声底**：同一份数据、只换 LightGBM 随机种子，RPS 会抖多少？
     任何小于这个抖动的 ΔRPS 都没有意义。--seeds N 就是用来量它的。

用法
----
    # 量噪声底：同一配置跑 5 个种子
    .venv/bin/python scripts/experiment_harness.py --seeds=5

    # 跑内置变体，与基线做配对比较
    .venv/bin/python scripts/experiment_harness.py --variants=no_odds,real_odds_only

    # 两者结合：每个变体各跑 3 个种子，用跨种子均值比较
    .venv/bin/python scripts/experiment_harness.py --variants=no_odds --seeds=3

    # 把 25/26 加为第 4 个测试季（features_v3 起可用；macro 从 15 格变 20 格）
    .venv/bin/python scripts/experiment_harness.py \
        --test-seasons=2022-2023,2023-2024,2024-2025,2025-2026 --seeds=5

自定义变体：在 VARIANTS 里加一项。transform 收到 (df, feature_cols)，返回
(新 df, 新 feature_cols)。**不允许增删行**——配对比较依赖测试集逐行对齐。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.lgbm_poisson import LGBMPoissonModel, FEATURE_COLS, LGB_PARAMS
from src.evaluation import match_outcome

# 默认 v3（21582 行，25/26 全季）。注意 features.csv 是「goals 已修、
# 赔率缺失未补」的中间版本（macro 0.1968，≈修 bug 前水平），别用它做基线。
FEATURES_PATH = ROOT / "data/opta/processed/features_v3.csv"
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {"EPL": "英超", "LL": "西甲", "SEA": "意甲", "BUN": "德甲", "LI1": "法甲"}
ODDS_COLS = ["odds_prob_home", "odds_prob_draw", "odds_prob_away"]


# ─── 变体定义 ────────────────────────────────────────────────────────────

def _drop_odds(df, cols):
    """整列删掉赔率特征（84 → 81），度量赔率的净贡献。"""
    return df, [c for c in cols if c not in ODDS_COLS]


def _real_odds_only(df, cols):
    """只保留有真实赔率的训练样本。

    注意这个变体会改变训练集大小，但**测试集不变**（只在 fit 前过滤），
    所以配对比较仍然成立。
    """
    df = df.copy()
    df["_has_odds"] = df["odds_prob_home"].notna()
    return df, cols


def _odds_as_nan(df, cols):
    """把赔率缺失显式保留为 NaN 而非中位数填充——LightGBM 原生支持 NaN 分裂。

    doc/tikaml_core_zh.md:331 声称「NaN 由 LightGBM 原生处理」，实际上
    lgbm_poisson.py:126-135 在推理前就把 NaN 全填成中位数了，模型从来没见过 NaN。
    这个变体测的是「如果真按文档说的做」会怎样。
    """
    return df, cols


VARIANTS = {
    "baseline": {"desc": "84 特征原样", "transform": None},
    "no_odds": {"desc": "删掉 odds_prob_* 三列 (81 特征)", "transform": _drop_odds},
    "real_odds_only": {"desc": "只用有真实赔率的样本训练", "transform": _real_odds_only,
                       "train_filter": "_has_odds"},
    "odds_as_nan": {"desc": "赔率缺失保留 NaN，不填中位数", "transform": _odds_as_nan,
                    "keep_nan": ODDS_COLS},
}


# ─── 核心 ────────────────────────────────────────────────────────────────

def rps_per_match(probs, outcomes):
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    cum_a = np.cumsum(onehot, axis=1)[:, :2]
    return ((cum_p - cum_a) ** 2).sum(axis=1) / 2.0


class _NoFillModel(LGBMPoissonModel):
    """保留指定列的 NaN，不做中位数填充。"""

    keep_nan = ()

    def _prepare_features(self, df):
        available = [c for c in self._feature_list if c in df.columns]
        self.feature_cols = available
        X = df[available].copy()
        if not hasattr(self, "_medians") or self._medians is None:
            self._medians = X.median()
        fill = self._medians.drop(labels=[c for c in self.keep_nan if c in self._medians.index])
        return X.fillna(fill)


def run_variant(df, name, spec, seed=None, temperature=0.90):
    """跑一个变体的完整 forward-chain，返回逐场 RPS 与对齐键。"""
    cols = list(FEATURE_COLS)
    work = spec.get("df", df)
    if spec.get("transform"):
        work, cols = spec["transform"](work, cols)

    params = LGB_PARAMS.copy()
    if seed is not None:
        params.update(random_state=seed, bagging_seed=seed,
                      feature_fraction_seed=seed, data_random_seed=seed)

    keys, rps_all, rows = [], [], []

    for test_season in TEST_SEASONS:
        all_train = work[work["season"] < test_season]
        lgbm_train = all_train[all_train["season"] >= "2016-2017"]
        prev = sorted(lgbm_train["season"].unique())[-1]
        val_data = lgbm_train[lgbm_train["season"] == prev]
        train_data = lgbm_train[lgbm_train["season"] < prev]

        if spec.get("train_filter"):
            f = spec["train_filter"]
            train_data = train_data[train_data[f]]
            val_data = val_data[val_data[f]]

        cls = _NoFillModel if spec.get("keep_nan") else LGBMPoissonModel
        model = cls(rho=-0.108, temperature=temperature,
                    params=params, feature_list=cols)
        if spec.get("keep_nan"):
            model.keep_nan = tuple(spec["keep_nan"])
        model.fit(train_data, val_df=val_data)

        for league in LEAGUES:
            test = work[(work["league"] == league) & (work["season"] == test_season)]
            if len(test) == 0:
                continue
            probs = model.predict_1x2(test)
            outcomes = np.array([match_outcome(r.home_goals, r.away_goals)
                                 for r in test.itertuples()])
            v = rps_per_match(probs, outcomes)
            rps_all.append(v)
            keys.extend(test.index.tolist())
            rows.append({"season": test_season, "league": league,
                         "n": len(test), "rps": float(v.mean())})

    v = np.concatenate(rps_all)
    res = pd.DataFrame(rows)
    return {
        "name": name, "seed": seed,
        "per_match": v, "keys": np.array(keys),
        "macro": float(res["rps"].mean()),          # 15 格算术平均（README 口径）
        "micro": float(v.mean()),                   # 逐场汇总
        "se": float(v.std(ddof=1) / np.sqrt(len(v))),
        "n": int(len(v)),
        "by_league": res.groupby("league")["rps"].mean().to_dict(),
    }


def paired_delta(base, variant):
    """配对 ΔRPS。测试场次必须逐行对齐，否则拒绝比较。"""
    if not np.array_equal(base["keys"], variant["keys"]):
        raise ValueError(f"{variant['name']} 的测试集与基线不一致，无法配对比较")
    d = variant["per_match"] - base["per_match"]
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"delta": float(d.mean()), "se": float(se),
            "significant": bool(abs(d.mean()) > 2 * se)}


def fmt_delta(dl):
    verdict = ("更差" if dl["delta"] > 0 else "更好") if dl["significant"] else "无法区分"
    return f"{dl['delta']:+.5f} ± {dl['se']:.5f}  → {verdict}"


def main():
    args = sys.argv[1:]
    seeds = 1
    want = ["baseline"]
    alt_features = None
    global TEST_SEASONS, FEATURES_PATH
    for a in args:
        if a.startswith("--seeds="):
            seeds = int(a.split("=", 1)[1])
        elif a.startswith("--variants="):
            want = ["baseline"] + [v for v in a.split("=", 1)[1].split(",") if v != "baseline"]
        elif a.startswith("--compare-features="):
            alt_features = Path(a.split("=", 1)[1])
        elif a.startswith("--test-seasons="):
            TEST_SEASONS = a.split("=", 1)[1].split(",")
        elif a.startswith("--features="):
            # 注意：默认的 features.csv 是 v2（25/26 只到 2026-03-09）。
            # 要用补齐的 25/26 必须显式指到 features_v3.csv。
            FEATURES_PATH = Path(a.split("=", 1)[1])
    unknown = [v for v in want if v not in VARIANTS]
    if unknown:
        sys.exit(f"未知变体 {unknown}，可选: {list(VARIANTS)}")

    t0 = time.time()
    print("=" * 78)
    print("TikaML 实验框架 — forward-chain 配对比较")
    print("=" * 78)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"], low_memory=False)
    # 25/26 只有在被指定为测试季时才保留（否则维持上一轮 22-25 口径）
    if "2025-2026" not in TEST_SEASONS:
        df = df[df["season"] != "2025-2026"].copy()

    # 另一份特征表作为变体参与比较。行数与行序必须与基线一致，否则无法配对。
    if alt_features:
        alt = pd.read_csv(alt_features, parse_dates=["date"], low_memory=False)
        if "2025-2026" not in TEST_SEASONS:
            alt = alt[alt["season"] != "2025-2026"].copy()
        if len(alt) != len(df) or not (alt.index == df.index).all():
            sys.exit(f"{alt_features} 的行数/行序与基线不一致，无法配对比较")
        diff_cols = [c for c in df.columns
                     if c in alt.columns and not df[c].equals(alt[c])]
        VARIANTS["alt_features"] = {
            "desc": f"另一份特征表 ({alt_features.name})，差异列: {diff_cols}",
            "transform": None, "df": alt,
        }
        want.append("alt_features")
        print(f"对比特征表: {alt_features}")
        print(f"  与基线不同的列 ({len(diff_cols)}): {diff_cols}")

    print(f"特征表 {len(df)} 场 | 测试季 {TEST_SEASONS} | 变体 {want} | 每个变体 {seeds} 个种子\n")

    seed_list = [None] if seeds == 1 else list(range(seeds))
    results = {}
    for name in want:
        spec = VARIANTS[name]
        runs = []
        for s in seed_list:
            r = run_variant(df, name, spec, seed=s)
            runs.append(r)
            tag = f" seed={s}" if s is not None else ""
            print(f"  {name:<16}{tag:<10} macro {r['macro']:.4f} | "
                  f"micro {r['micro']:.4f} ± {r['se']:.4f} ({r['n']} 场)")
        results[name] = runs

    # 噪声底：同一配置跨种子的抖动
    if seeds > 1:
        print(f"\n{'─' * 78}\n噪声底（同一配置，只换随机种子）\n{'─' * 78}")
        for name, runs in results.items():
            macros = np.array([r["macro"] for r in runs])
            micros = np.array([r["micro"] for r in runs])
            print(f"  {name:<16} macro {macros.mean():.4f} ± {macros.std(ddof=1):.5f} (跨种子标准差) "
                  f"| 极差 {macros.max() - macros.min():.5f}")
            print(f"  {'':<16} micro {micros.mean():.4f} ± {micros.std(ddof=1):.5f}")
        base_macros = np.array([r["macro"] for r in results[want[0]]])
        floor = base_macros.std(ddof=1)
        print(f"\n  → 任何小于 ~{2 * floor:.5f} (2×跨种子标准差) 的 ΔRPS 都无法与训练随机性区分。")

    # 配对比较
    if len(want) > 1:
        print(f"\n{'─' * 78}\n与基线的配对比较\n{'─' * 78}")
        base = results[want[0]]
        for name in want[1:]:
            deltas = []
            for i, r in enumerate(results[name]):
                dl = paired_delta(base[min(i, len(base) - 1)], r)
                deltas.append(dl)
            if len(deltas) == 1:
                print(f"  {name:<16} {VARIANTS[name]['desc']}")
                print(f"  {'':<16} ΔRPS {fmt_delta(deltas[0])}")
            else:
                ds = np.array([d["delta"] for d in deltas])
                print(f"  {name:<16} {VARIANTS[name]['desc']}")
                print(f"  {'':<16} ΔRPS {ds.mean():+.5f} ± {ds.std(ddof=1):.5f} (跨种子) | "
                      f"各种子 {', '.join(f'{d:+.5f}' for d in ds)}")

    out = {n: [{k: v for k, v in r.items() if k not in ("per_match", "keys")} for r in runs]
           for n, runs in results.items()}
    dest = ROOT / "experiments/results/experiment_results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n耗时 {time.time() - t0:.0f}s，结果写入 {dest}")


if __name__ == "__main__":
    main()
