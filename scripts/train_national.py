"""Train the national-team goal model on ALL international match data.

Fits time-weighted neutral-aware Poisson strength ratings, prints the strength
table for a sanity check, saves the model, and demos World Cup 2026 fixtures.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.national_poisson import NationalTeamModel

DATA = "data/international/international_results.csv"
WC_START = "2026-06-11"


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    played = df[df.home_score.notna()]
    print(f"全部数据: {len(df)} 场, 已完赛 {len(played)} 场, "
          f"{played.date.min().date()} → {played.date.max().date()}")

    model = NationalTeamModel()
    model.fit(df, ref_date=WC_START)
    print(f"\n拟合完成: {len(model.attack)} 队, home_adv={model.home_adv:.3f} "
          f"(≈{(2.718**model.home_adv - 1) * 100:.0f}% 主场进球加成), rho={model.rho:.3f}, "
          f"半衰期={model.half_life_days}天, ref_date={model.ref_date.date()}")

    print("\n=== 实力榜 TOP20 (合理性检验) ===")
    top = model.team_ratings(20)
    for r in top.itertuples():
        print(f"  {r.Index + 1:2d}. {r.team:<22} strength={r.strength:+.3f} "
              f"(att={r.attack:+.2f} def={r.defense:+.2f})")

    model.save("models/national")

    # WC2026 示范预测
    wc = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
    print(f"\n=== WC2026 赛程示范预测 (前5场) ===")
    for r in wc.head(5).itertuples():
        res = model.predict(r.home_team, r.away_team, neutral=bool(r.neutral))
        p = res["probs_1x2"]
        ou25 = res["goals_over_under"][2.5]
        print(f"\n  {r.home_team} vs {r.away_team}  "
              f"({'中立' if r.neutral else '主场'} @ {r.country})")
        print(f"    λ: {res['lambda_home']:.2f} - {res['lambda_away']:.2f}  | "
              f"1X2: 主 {p[0]:.0%} / 平 {p[1]:.0%} / 客 {p[2]:.0%}")
        print(f"    推荐比分: {res['recommended_score']['label']} "
              f"({res['recommended_score']['prob']:.0%})  | "
              f"大2.5球: {ou25['over']:.0%} / 小: {ou25['under']:.0%}")


if __name__ == "__main__":
    main()
