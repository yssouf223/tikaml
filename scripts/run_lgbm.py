"""Phase 2: LightGBM Poisson model evaluation with forward-chain validation.

Compares LightGBM against Dixon-Coles baseline.

Usage:
    .venv/bin/python scripts/run_lgbm.py                # 含 Dixon-Coles 对照
    .venv/bin/python scripts/run_lgbm.py --no-dc        # 只跑 LightGBM（快）
    .venv/bin/python scripts/run_lgbm.py --features=... # 指定特征表

训练依赖见 requirements-train.txt（lgb.LGBMRegressor 需要 scikit-learn，
requirements-server.txt 只覆盖 Booster 推理路径）。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.lgbm_poisson import LGBMPoissonModel
from src.evaluation import evaluate_predictions, match_outcome

FEATURES_PATH = ROOT / "data/opta/processed/features.csv"
TEST_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
LEAGUES = ["EPL", "LL", "SEA", "BUN", "LI1"]
LEAGUE_NAMES = {
    "EPL": "英超", "LL": "西甲", "SEA": "意甲",
    "BUN": "德甲", "LI1": "法甲",
}


def load_dixon_coles():
    """Dixon-Coles 基线在 experiments/src/ 下，与 src/ 同名包冲突，按路径加载。"""
    path = ROOT / "experiments/src/dixon_coles.py"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，用 --no-dc 跳过 Dixon-Coles 对照")
    spec = importlib.util.spec_from_file_location("exp_dixon_coles", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DixonColesModel


def rps_per_match(probs, outcomes):
    """逐场 RPS 向量，用于计算标准误。"""
    cum_p = np.cumsum(probs, axis=1)[:, :2]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    cum_a = np.cumsum(onehot, axis=1)[:, :2]
    return ((cum_p - cum_a) ** 2).sum(axis=1) / 2.0


def run_validation(features_path=FEATURES_PATH, with_dc=True):
    print("=" * 70)
    print("TikaML Phase 2: LightGBM Poisson 模型验证")
    print("=" * 70)

    # Load feature table
    print("\n加载特征表...")
    df = pd.read_csv(features_path, parse_dates=["date"], low_memory=False)
    df = df[df["season"] != "2025-2026"].copy()
    print(f"  {len(df)} 场比赛, {len(df.columns)} 列")

    # Check feature availability
    from src.lgbm_poisson import FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    print(f"  可用特征: {len(available)}/{len(FEATURE_COLS)}")
    if missing:
        print(f"  缺失特征: {missing}")

    DixonColesModel = load_dixon_coles() if with_dc else None

    all_results = []
    pooled_lgbm, pooled_dc = [], []

    for test_season in TEST_SEASONS:
        print(f"\n{'─' * 70}")
        print(f"测试赛季: {test_season}")
        print(f"{'─' * 70}")

        # Train ONE LightGBM model on ALL leagues (cross-league training)
        all_train = df[df["season"] < test_season].copy()
        all_train_lgbm = all_train[all_train["season"] >= "2016-2017"].copy()

        # Use last season before test as validation for early stopping
        prev_season = sorted(all_train_lgbm["season"].unique())[-1]
        val_data = all_train_lgbm[all_train_lgbm["season"] == prev_season]
        train_data = all_train_lgbm[all_train_lgbm["season"] < prev_season]

        lgbm = LGBMPoissonModel(rho=-0.108)
        lgbm.fit(train_data, val_df=val_data)

        for league in LEAGUES:
            league_df = df[df["league"] == league].copy()
            train = league_df[league_df["season"] < test_season]
            test = league_df[league_df["season"] == test_season]

            if len(test) == 0:
                continue

            # LightGBM predictions (model already trained)
            lgbm_probs = lgbm.predict_1x2(test)

            # Outcomes
            outcomes = np.array([
                match_outcome(r["home_goals"], r["away_goals"])
                for _, r in test.iterrows()
            ])

            lgbm_metrics = evaluate_predictions(lgbm_probs, outcomes)
            pooled_lgbm.append(rps_per_match(lgbm_probs, outcomes))

            name = LEAGUE_NAMES.get(league, league)
            print(f"\n  {name} ({league}) — {len(test)} 场")
            print(f"    {'模型':<18} {'RPS':>8} {'Brier':>8} {'LogLoss':>8} {'准确率':>8}")
            print(f"    {'LightGBM':<18} {lgbm_metrics['rps']:>8.4f} "
                  f"{lgbm_metrics['brier']:>8.4f} {lgbm_metrics['log_loss']:>8.4f} "
                  f"{lgbm_metrics['accuracy']:>7.1%}")

            row = {
                "season": test_season,
                "league": league,
                "n_matches": lgbm_metrics["n_matches"],
                "rps_lgbm": lgbm_metrics["rps"],
                "acc_lgbm": lgbm_metrics["accuracy"],
                "brier_lgbm": lgbm_metrics["brier"],
            }

            if with_dc:
                # Dixon-Coles baseline (for comparison)
                dc = DixonColesModel(half_life_days=180)
                test_start = test["date"].min()
                dc.fit(train[["date", "home_team", "away_team",
                              "home_goals", "away_goals"]],
                       current_date=test_start)

                dc_probs = []
                for _, r in test.iterrows():
                    try:
                        p = dc.predict_1x2(r["home_team"], r["away_team"])
                        if np.any(np.isnan(p)):
                            p = np.array([0.40, 0.30, 0.30])
                    except (KeyError, ValueError):
                        p = np.array([0.40, 0.30, 0.30])
                    dc_probs.append(p)
                dc_probs = np.array(dc_probs)

                dc_metrics = evaluate_predictions(dc_probs, outcomes)
                pooled_dc.append(rps_per_match(dc_probs, outcomes))
                print(f"    {'Dixon-Coles':<18} {dc_metrics['rps']:>8.4f} "
                      f"{dc_metrics['brier']:>8.4f} {dc_metrics['log_loss']:>8.4f} "
                      f"{dc_metrics['accuracy']:>7.1%}")

                delta_rps = lgbm_metrics["rps"] - dc_metrics["rps"]
                print(f"    → LightGBM vs DC: RPS {delta_rps:+.4f} "
                      f"({'改善' if delta_rps < 0 else '退步'})")
                row["rps_dc"] = dc_metrics["rps"]
                row["acc_dc"] = dc_metrics["accuracy"]

            all_results.append(row)

        # Print feature importance for the last model
        print(f"\n  特征重要性 (最后一个模型):")
        fi = lgbm.feature_importance(top_n=15)
        for _, r in fi.iterrows():
            print(f"    {r['feature']:<40} {r['importance_avg']:>6.0f}")

    # Summary
    results_df = pd.DataFrame(all_results)
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")

    print(f"\n各联赛平均 RPS:")
    header = f"  {'联赛':<6} {'LightGBM':>10}"
    if with_dc:
        header += f" {'Dixon-Coles':>12} {'差值':>8}"
    print(header + f" {'准确率':>8}")
    for league in LEAGUES:
        lg = results_df[results_df["league"] == league]
        if len(lg) == 0:
            continue
        name = LEAGUE_NAMES.get(league, league)
        line = f"  {name:<6} {lg['rps_lgbm'].mean():>10.4f}"
        if with_dc:
            line += (f" {lg['rps_dc'].mean():>12.4f} "
                     f"{lg['rps_lgbm'].mean() - lg['rps_dc'].mean():>+8.4f}")
        print(line + f" {lg['acc_lgbm'].mean():>7.1%}")

    overall_lgbm = results_df["rps_lgbm"].mean()
    line = f"\n  {'总计':<6} {overall_lgbm:>10.4f}"
    if with_dc:
        overall_dc = results_df["rps_dc"].mean()
        line += (f" {overall_dc:>12.4f} {overall_lgbm - overall_dc:>+8.4f}")
    print(line + f" {results_df['acc_lgbm'].mean():>7.1%}")
    print(f"  （以上为 {len(results_df)} 个「联赛×赛季」格子的算术平均）")

    # 逐场汇总 + 标准误：ΔRPS 小于 2×SE 一律视为无法区分
    v = np.concatenate(pooled_lgbm)
    n = len(v)
    print(f"\n逐场汇总 ({n} 场):")
    print(f"  LightGBM     RPS {v.mean():.4f} ± {v.std(ddof=1) / np.sqrt(n):.4f} (标准误)")
    if with_dc:
        vdc = np.concatenate(pooled_dc)
        d = vdc - v
        se_d = d.std(ddof=1) / np.sqrt(n)
        print(f"  Dixon-Coles  RPS {vdc.mean():.4f} ± {vdc.std(ddof=1) / np.sqrt(n):.4f}")
        print(f"  配对差 DC − LGBM = {d.mean():+.5f} ± {se_d:.5f}  "
              f"({'显著' if abs(d.mean()) > 2 * se_d else '无法区分'})")

    print(f"\n参考基准:")
    print(f"  Dixon-Coles: ~0.205")
    print(f"  本方案目标:  ~0.190-0.198")
    print(f"  博彩公司:    ~0.185")


if __name__ == "__main__":
    features = FEATURES_PATH
    for arg in sys.argv[1:]:
        if arg.startswith("--features="):
            features = Path(arg.split("=", 1)[1])
    run_validation(features_path=features, with_dc="--no-dc" not in sys.argv)
