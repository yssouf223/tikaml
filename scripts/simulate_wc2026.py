"""Demo: World Cup 2026 tournament simulation + live in-match prediction."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.national_poisson import NationalTeamModel
from src.national_simulate import TournamentSimulator, derive_groups
from src.national_live import live_predict

DATA = "data/international/international_results.csv"


def main():
    model = NationalTeamModel.load("models/national")
    df = pd.read_csv(DATA, parse_dates=["date"])
    fixtures = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == 2026)]
    groups, group_matches = derive_groups(fixtures)

    sim = TournamentSimulator(model)
    print("\n模拟 WC2026 (10000 次)...")
    table = sim.simulate(groups, group_matches, n_sims=10000, seed=1)

    print(f"\n{'='*70}\n夺冠概率 TOP15\n{'='*70}")
    print(f"{'队伍':<22}{'组':<4}{'出线':<8}{'8强':<8}{'4强':<8}{'决赛':<8}{'夺冠'}")
    print("-" * 70)
    for r in table.head(15).itertuples():
        print(f"{r.team:<22}{r.group:<4}{r.P_advance:<8.0%}{r.P_QF:<8.0%}"
              f"{r.P_SF:<8.0%}{r.P_final:<8.0%}{r.P_champion:.1%}")

    print(f"\n{'='*70}\n各组出线概率\n{'='*70}")
    for gl in sorted(groups):
        sub = table[table.group == gl].sort_values("P_advance", ascending=False)
        line = f"  组{gl}: " + " | ".join(
            f"{r.team} {r.P_advance:.0%}" for r in sub.itertuples())
        print(line)

    # ---- 滚球演示 ----
    print(f"\n{'='*70}\n滚球演示: Argentina vs Algeria (中立场, 假设进程)\n{'='*70}")
    lh, la = model.predict_lambdas("Argentina", "Algeria", neutral=True)
    print(f"  赛前 λ: {lh:.2f} - {la:.2f}")
    scenarios = [
        (0, 0, 0, 0, 0, "开场"),
        (25, 1, 0, 0, 0, "25' 阿根廷进球 1-0"),
        (55, 1, 0, 1, 0, "55' 阿根廷染红 (10人)"),
        (70, 1, 1, 1, 0, "70' 阿尔及利亚扳平 1-1"),
        (88, 1, 1, 1, 0, "88' 仍 1-1"),
    ]
    for minute, hg, ag, hr, ar, label in scenarios:
        out = live_predict(model, "Argentina", "Algeria", True, minute, hg, ag,
                           home_red_cards=hr, away_red_cards=ar,
                           lambda_home=lh, lambda_away=la)
        p = out["probs_1x2"]
        lhr, lar = out["lambda_remaining"]
        print(f"  {label:<28} 1X2: 阿 {p[0]:.0%} / 平 {p[1]:.0%} / 阿尔及 {p[2]:.0%}"
              f"  (剩余λ {lhr:.2f}-{lar:.2f})")


if __name__ == "__main__":
    main()
