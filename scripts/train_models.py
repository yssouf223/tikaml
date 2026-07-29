"""训练并落盘生产模型（goals / corners / yellows）。

此前仓库里没有这个入口：goals 模型只能靠 src/inference.py 的 MatchPredictor.train(save=True)
产出，corners / yellows 的训练脚本则从未提交（commit 91daf1d 只提交了模型产物）。
结果是 models/ 下的模型无法被重新生成。本脚本补上这个缺口。

训练口径直接复用 MatchPredictor.train()，保证与推理侧同源：
  - min_season 之后的全部数据（默认 2016-2017，即剔除 2014-15/2015-16 两个滚动特征预热季）
  - 按 date 排序后前 80% 训练、后 20% 作 early stopping 验证
  - goals: 84 特征 / corners: 89 / yellows: 91

默认**不写盘**（models/ 下是生产模型，避免误覆盖），要落盘必须显式 --save。

用法:
    .venv/bin/python scripts/train_models.py                          # 只训练，打印摘要
    .venv/bin/python scripts/train_models.py --save --out=models/cand_20260727
    .venv/bin/python scripts/train_models.py --save                   # 覆盖 models/（危险）
    .venv/bin/python scripts/train_models.py --min-season=2018-2019 --goals-only

依赖见 requirements-train.txt。
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.inference import MatchPredictor


def medians_fingerprint(model):
    """medians 的稳定指纹——用于日后判定某个模型产物出自哪份特征表。

    做溯源时发现 models/meta.json 的 84 个 medians 可以唯一确定训练口径
    （特征表版本 + min_season + 80/20 切分），这个指纹让以后不必再逐一比对。
    """
    payload = json.dumps(
        {k: round(float(v), 10) for k, v in sorted(model._medians.to_dict().items())},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def describe(name, model, n_train, n_val):
    if model is None:
        print(f"  {name:<8} 跳过（样本不足）")
        return
    bi_h = getattr(model.model_home, "best_iteration_", None)
    bi_a = getattr(model.model_away, "best_iteration_", None)
    print(f"  {name:<8} {len(model.feature_cols):>3} 特征 | "
          f"训练 {n_train} / 验证 {n_val} | "
          f"best_iter home={bi_h} away={bi_a} | "
          f"medians 指纹 {medians_fingerprint(model)}")


def main():
    args = sys.argv[1:]
    save = "--save" in args
    goals_only = "--goals-only" in args
    min_season = "2016-2017"
    out_dir = ROOT / "models"
    features_path = ROOT / "data/opta/processed/features.csv"

    for a in args:
        if a.startswith("--min-season="):
            min_season = a.split("=", 1)[1]
        elif a.startswith("--out="):
            p = Path(a.split("=", 1)[1])
            out_dir = p if p.is_absolute() else ROOT / p
        elif a.startswith("--features="):
            p = Path(a.split("=", 1)[1])
            features_path = p if p.is_absolute() else ROOT / p

    print("=" * 70)
    print("TikaML 模型训练")
    print("=" * 70)
    print(f"特征表    : {features_path}")
    print(f"min_season: {min_season}")
    print(f"输出      : {out_dir if save else '(不写盘，加 --save 才落盘)'}")

    predictor = MatchPredictor(features_path=features_path)
    predictor.load_data()
    all_data = predictor.df[predictor.df["season"] >= min_season]
    n = len(all_data)
    split = int(n * 0.8)
    print(f"训练样本  : {n} 场 "
          f"({all_data['date'].min():%Y-%m-%d} → {all_data['date'].max():%Y-%m-%d}), "
          f"前 80% = {split} 训练 / {n - split} 验证\n")

    predictor.train(min_season=min_season, save=False)

    print("\n训练结果:")
    describe("goals", predictor.model, split, n - split)
    if not goals_only:
        describe("corners", predictor.corner_model, "—", "—")
        describe("yellows", predictor.yellow_model, "—", "—")

    if not save:
        print("\n未落盘。确认无误后加 --save（建议同时用 --out= 指向新目录，"
              "不要直接覆盖 models/ 下的生产模型）。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n落盘到 {out_dir}/")
    predictor.model.save(str(out_dir))
    if not goals_only:
        if predictor.corner_model is not None:
            predictor.corner_model.save(str(out_dir / "corners"))
        if predictor.yellow_model is not None:
            predictor.yellow_model.save(str(out_dir / "yellows"))

    manifest = {
        "features_path": str(features_path),
        "features_mtime": features_path.stat().st_mtime,
        "min_season": min_season,
        "n_matches": n,
        "n_train": split,
        "n_val": n - split,
        "date_range": [str(all_data["date"].min().date()),
                       str(all_data["date"].max().date())],
        "medians_fingerprint": {
            "goals": medians_fingerprint(predictor.model),
        },
    }
    if not goals_only:
        if predictor.corner_model is not None:
            manifest["medians_fingerprint"]["corners"] = medians_fingerprint(predictor.corner_model)
        if predictor.yellow_model is not None:
            manifest["medians_fingerprint"]["yellows"] = medians_fingerprint(predictor.yellow_model)
    (out_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  训练清单已写入 {out_dir}/train_manifest.json")


if __name__ == "__main__":
    main()
