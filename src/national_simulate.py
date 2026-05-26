"""Monte-Carlo tournament simulation for the 2026 World Cup (groups -> knockouts).

Simulates the group stage from the model's match probabilities to get real
standings (points / GD / GF tiebreakers), determines the top-2 per group plus
the 8 best third-placed teams, then runs FIFA's official single-elimination
knockout bracket (Round of 32 -> Final) to estimate each team's advancement and
title probabilities. Knockout draws are resolved by a ~50/50 penalty shootout.

Knockout pairings follow FIFA's official 2026 bracket template (match numbers
73-104): the 24 group winners / runners-up occupy fixed slots keyed by their
official group letter, and the 8 best thirds are slotted via FIFA's eligibility
lists (each third-slot accepts a third from one of 5 specific groups; group K's
third can only go to one slot, group L's to one other). When more than one
assignment satisfies the eligibility lists, a constraint-respecting bipartite
matching is used: it is always a FIFA-legal bracket and differs from FIFA's
exact 495-row lookup table only in the rare multi-solution case, with negligible
effect on title odds since thirds are the weakest qualifiers.

Supports two kinds of in-tournament dynamics:
  - `played`: fix already-played group-match scores; only remaining group
    matches are simulated. This re-ranks groups / re-fills the bracket as real
    results land.
  - re-fitting the strength model on new results before simulating (caller
    passes an updated model) — the "re-ranking" of team strength mid-tournament.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


# Official 2026 final-draw group composition (drawn 2025-12-05). This is the
# source of truth for group letters; the fixtures only supply the match-ups and
# neutral-venue flags. Team names match data/international/international_results.csv.
OFFICIAL_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
_TEAM_GROUP = {t: gl for gl, ts in OFFICIAL_GROUPS.items() for t in ts}

# Official Round of 32 template (FIFA match numbers 73-88). Each match is two
# slots; a slot is a group winner ("W"), runner-up ("R"), or a best-third
# placeholder ("3", <eligible groups>) resolved per simulation by _match_thirds.
R32 = [
    (73, ("R", "A"), ("R", "B")),
    (74, ("W", "E"), ("3", "ABCDF")),
    (75, ("W", "F"), ("R", "C")),
    (76, ("W", "C"), ("R", "F")),
    (77, ("W", "I"), ("3", "CDFGH")),
    (78, ("R", "E"), ("R", "I")),
    (79, ("W", "A"), ("3", "CEFHI")),
    (80, ("W", "L"), ("3", "EHIJK")),
    (81, ("W", "D"), ("3", "BEFIJ")),
    (82, ("W", "G"), ("3", "AEHIJ")),
    (83, ("R", "K"), ("R", "L")),
    (84, ("W", "H"), ("R", "J")),
    (85, ("W", "B"), ("3", "EFGIJ")),
    (86, ("W", "J"), ("R", "H")),
    (87, ("W", "K"), ("3", "DEIJL")),
    (88, ("R", "D"), ("R", "G")),
]
# Later rounds: match -> (feeder match 1, feeder match 2). Winners advance.
R16 = {89: (73, 75), 90: (74, 77), 91: (76, 78), 92: (79, 80),
       93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF = {101: (97, 98), 102: (99, 100)}
FINAL = {104: (101, 102)}

# Third-place slots in match order, each with its eligible source groups.
_THIRD_SLOTS = [(m, set(elig)) for (m, s1, s2) in R32
                for (kind, elig) in (s1, s2) if kind == "3"]


def _match_thirds(third_groups, rng):
    """Slot the 8 qualifying third-place groups into the 8 third-slots.

    Returns {group_letter: match_number}. Uses Kuhn's bipartite matching over
    FIFA's eligibility lists; a perfect matching always exists because FIFA's
    table covers all C(12,8)=495 combinations. Group order is permuted per
    simulation so the (seed-reproducible) choice among multiple legal matchings
    is unbiased rather than systematically favouring one slotting.
    """
    groups = list(third_groups)
    groups = [groups[i] for i in rng.permutation(len(groups))]
    slot_for = {}                                    # match_number -> group

    def assign(g, seen):
        for (m, elig) in _THIRD_SLOTS:
            if g in elig and m not in seen:
                seen.add(m)
                if m not in slot_for or assign(slot_for[m], seen):
                    slot_for[m] = g
                    return True
        return False

    for g in groups:
        assign(g, set())
    assert len(slot_for) == 8, f"no legal third slotting for {sorted(third_groups)}"
    return {g: m for m, g in slot_for.items()}


def derive_groups(fixtures):
    """Build official-lettered groups (A-L) and the group-stage match list.

    Group letters come from OFFICIAL_GROUPS (the final-draw composition); the
    fixtures supply the actual match-ups and neutral-venue flags. Asserts every
    fixture team is a known 2026 participant so a stale data refresh fails loudly
    instead of mis-slotting the bracket.
    """
    teams = set(fixtures.home_team) | set(fixtures.away_team)
    unknown = teams - set(_TEAM_GROUP)
    assert not unknown, f"teams not in official 2026 groups: {sorted(unknown)}"
    groups = {gl: [t for t in ts if t in teams] for gl, ts in OFFICIAL_GROUPS.items()}
    for gl, ts in groups.items():
        assert len(ts) == 4, f"group {gl}: {len(ts)} teams present in fixtures, expected 4"
    matches = [(r.home_team, r.away_team, bool(r.neutral)) for r in fixtures.itertuples()]
    return groups, matches


def derive_bracket(groups, played, seed=0):
    """Resolve the 16 Round-of-32 match-ups from a COMPLETED group stage.

    `played`: {(home, away): (hg, ag)} covering every group match. Returns
    {match_no: (team1, team2)} for the R32 matches, applying the official
    template + one legal best-third slotting. Feed the result to
    simulate_from_bracket. (For the real tournament you may instead pass FIFA's
    actual R32 draw directly — they agree up to the rare multi-solution third
    slotting.)"""
    rng = np.random.default_rng(seed)
    team_group = {t: gl for gl, ts in groups.items() for t in ts}
    pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
    games = defaultdict(int)
    for (h, a), (hg, ag) in played.items():
        if team_group.get(h) != team_group.get(a):       # ignore non-group matches
            continue
        games[h] += 1; games[a] += 1
        gf[h] += hg; gf[a] += ag; gd[h] += hg - ag; gd[a] += ag - hg
        if hg > ag: pts[h] += 3
        elif hg < ag: pts[a] += 3
        else: pts[h] += 1; pts[a] += 1
    # guard: each team must have played all 3 group games, else standings (and
    # thus the derived bracket) would be silently wrong
    incomplete = [t for t in team_group if games[t] != 3]
    assert not incomplete, \
        f"group stage incomplete ({len(incomplete)} teams without 3 games): {sorted(incomplete)[:5]}"

    def key(t): return (pts[t], gd[t], gf[t], rng.random())
    winners, runners, thirds = {}, {}, []
    for gl, ts in groups.items():
        ranked = sorted(ts, key=key, reverse=True)
        assert len(ranked) == 4, f"group {gl} must have 4 teams"
        winners[gl] = ranked[0]; runners[gl] = ranked[1]; thirds.append((gl, ranked[2]))
    best = sorted(thirds, key=lambda x: (pts[x[1]], gd[x[1]], gf[x[1]], rng.random()),
                  reverse=True)[:8]
    third_team = {gl: t for gl, t in best}
    third_slot = _match_thirds([gl for gl, _ in best], rng)
    third_at = {m: third_team[gl] for gl, m in third_slot.items()}
    bracket = {}
    for (m, s1, s2) in R32:
        pair = []
        for slot in (s1, s2):
            if slot[0] == "W": pair.append(winners[slot[1]])
            elif slot[0] == "R": pair.append(runners[slot[1]])
            else: pair.append(third_at[m])
        bracket[m] = (pair[0], pair[1])
    return bracket


class TournamentSimulator:

    def __init__(self, model):
        self.model = model
        self._cache = {}

    def match_probs(self, home, away, neutral):
        """(p_home, p_draw, p_away) from the model's score matrix."""
        key = (home, away, neutral)
        if key not in self._cache:
            lh, la = self.model.predict_lambdas(home, away, neutral)
            m = self.model.predict_score_matrix(lh, la)
            self._cache[key] = (float(np.tril(m, -1).sum()), float(np.trace(m)),
                                float(np.triu(m, 1).sum()))
        return self._cache[key]

    def _sample_score(self, home, away, neutral, rng):
        lh, la = self.model.predict_lambdas(home, away, neutral)
        return rng.poisson(lh), rng.poisson(la)

    def _winner(self, t1, t2, rng):
        """Knockout winner; a draw is resolved by a ~50/50 penalty shootout."""
        p1, pdr, p2 = self.match_probs(t1, t2, True)
        p1_adv = p1 + 0.5 * pdr
        return t1 if rng.random() < p1_adv / (p1_adv + p2 + 0.5 * pdr) else t2

    def _pin_or_play(self, m, t1, t2, played_ko, rng):
        """Return the pinned winner of match `m` if given (and a participant),
        otherwise simulate it."""
        if played_ko and m in played_ko and played_ko[m] in (t1, t2):
            return played_ko[m]
        return self._winner(t1, t2, rng)

    def _play_knockout(self, r32_pairs, played_ko, rng, reached):
        """Play R32 -> Final from the 16 R32 match-ups, honouring played_ko pins.
        r32_pairs: {match_no: (team1, team2)}; updates the `reached` counters."""
        winner_of = {}
        for (m, _s1, _s2) in R32:
            t1, t2 = r32_pairs[m]
            winner_of[m] = self._pin_or_play(m, t1, t2, played_ko, rng)
        for m in winner_of:                       # R32 winners reach the R16
            reached["R16"][winner_of[m]] += 1
        for feed, label in [(R16, "QF"), (QF, "SF"), (SF, "F"), (FINAL, "champion")]:
            for m, (a, b) in feed.items():
                w = self._pin_or_play(m, winner_of[a], winner_of[b], played_ko, rng)
                winner_of[m] = w
                reached[label][w] += 1

    def simulate(self, groups, group_matches, n_sims=5000, played=None,
                 played_ko=None, seed=0):
        rng = np.random.default_rng(seed)
        played = played or {}
        played_ko = played_ko or {}
        assert set(groups) == set(OFFICIAL_GROUPS), \
            f"expected official groups A-L, got {sorted(groups)}"

        teams = [t for g in groups.values() for t in g]
        team_group = {t: gl for gl, ts in groups.items() for t in ts}

        adv = defaultdict(int)
        won_group = defaultdict(int)
        reached = {r: defaultdict(int) for r in ["R16", "QF", "SF", "F", "champion"]}

        for _ in range(n_sims):
            # ---- group stage ----
            pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
            for (h, a, neut) in group_matches:
                if (h, a) in played:
                    hg, ag = played[(h, a)]
                else:
                    hg, ag = self._sample_score(h, a, neut, rng)
                gf[h] += hg; gf[a] += ag; gd[h] += hg - ag; gd[a] += ag - hg
                if hg > ag: pts[h] += 3
                elif hg < ag: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1

            def key(t): return (pts[t], gd[t], gf[t], rng.random())
            winners, runners, thirds = {}, {}, []
            for gl, ts in groups.items():
                ranked = sorted(ts, key=key, reverse=True)
                winners[gl] = ranked[0]; runners[gl] = ranked[1]
                thirds.append((gl, ranked[2]))
                won_group[ranked[0]] += 1
            # best 8 of the 12 third-placed teams
            best = sorted(thirds, key=lambda x: (pts[x[1]], gd[x[1]], gf[x[1]], rng.random()),
                          reverse=True)[:8]
            third_team = {gl: t for gl, t in best}
            for t in list(winners.values()) + list(runners.values()) + [t for _, t in best]:
                adv[t] += 1

            # ---- knockout: official FIFA bracket ----
            third_slot = _match_thirds([gl for gl, _ in best], rng)   # group -> match_no
            third_at = {m: third_team[gl] for gl, m in third_slot.items()}
            r32_pairs = {}
            for (m, s1, s2) in R32:
                pair = []
                for slot in (s1, s2):
                    if slot[0] == "W": pair.append(winners[slot[1]])
                    elif slot[0] == "R": pair.append(runners[slot[1]])
                    else: pair.append(third_at[m])
                r32_pairs[m] = (pair[0], pair[1])
            self._play_knockout(r32_pairs, played_ko, rng, reached)

        rows = []
        for t in teams:
            rows.append({
                "team": t, "group": team_group[t],
                "P_advance": adv[t] / n_sims,
                "P_win_group": won_group[t] / n_sims,
                "P_R16": reached["R16"][t] / n_sims,
                "P_QF": reached["QF"][t] / n_sims,
                "P_SF": reached["SF"][t] / n_sims,
                "P_final": reached["F"][t] / n_sims,
                "P_champion": reached["champion"][t] / n_sims,
            })
        return pd.DataFrame(rows).sort_values("P_champion", ascending=False).reset_index(drop=True)

    def simulate_from_bracket(self, bracket, played_ko=None, n_sims=10000, seed=0):
        """Round-reach / title probabilities from a FIXED Round-of-32 bracket.

        bracket: {match_no: (team1, team2)} for the 16 R32 matches (73-88).
        played_ko: {match_no: winner} for knockout matches already decided.

        Use once the group stage is over (the 32 qualifiers and their slots are
        known, e.g. via derive_bracket) to recompute title odds after every
        knockout result — eliminated teams drop to 0, survivors re-concentrate."""
        rng = np.random.default_rng(seed)
        played_ko = played_ko or {}
        assert set(bracket) == {m for (m, _, _) in R32}, \
            "bracket must specify all 16 R32 matches (numbers 73-88)"
        teams = [t for pair in bracket.values() for t in pair]
        reached = {r: defaultdict(int) for r in ["R16", "QF", "SF", "F", "champion"]}
        for _ in range(n_sims):
            self._play_knockout(bracket, played_ko, rng, reached)
        rows = [{
            "team": t,
            "P_R16": reached["R16"][t] / n_sims,
            "P_QF": reached["QF"][t] / n_sims,
            "P_SF": reached["SF"][t] / n_sims,
            "P_final": reached["F"][t] / n_sims,
            "P_champion": reached["champion"][t] / n_sims,
        } for t in teams]
        return pd.DataFrame(rows).sort_values("P_champion", ascending=False).reset_index(drop=True)
