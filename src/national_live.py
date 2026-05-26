"""Live in-match prediction for national teams.

The national strength model provides pre-match lambda_home / lambda_away;
those feed the existing Bayesian LivePredictor, which updates the remaining-time
scoring rates after each goal / red card (non-linear time decay, score momentum,
red-card factors). This is the in-match ("滚球") layer for the World Cup model.
"""

from src.live_predictor import LivePredictor


def live_predict(model, home_team, away_team, neutral, minute,
                 home_goals, away_goals, home_red_cards=0, away_red_cards=0,
                 lambda_home=None, lambda_away=None):
    """Update the live prediction given the current match state.

    Call again after every goal / red card (or every minute) with the new
    minute + score + red-card counts to get refreshed probabilities.

    Args:
        model: a fitted NationalTeamModel (provides pre-match lambda + rho).
        home_team, away_team, neutral: the fixture (neutral=True for non-host WC games).
        minute, home_goals, away_goals: current match state.
        home_red_cards, away_red_cards: red cards so far.
        lambda_home, lambda_away: optional pre-computed pre-match lambdas
            (pass them to avoid recomputing across repeated live calls).

    Returns: live probability dict (see LivePredictor.get_probabilities) plus
        the pre-match lambdas used.
    """
    if lambda_home is None or lambda_away is None:
        lambda_home, lambda_away = model.predict_lambdas(home_team, away_team, neutral)
    lp = LivePredictor(lambda_home, lambda_away,
                       rho=(model.rho if model.rho is not None else -0.10))
    lp.update(minute=minute, home_goals=home_goals, away_goals=away_goals,
              home_red_cards=home_red_cards, away_red_cards=away_red_cards)
    out = lp.get_probabilities()
    out["lambda_prematch"] = (lambda_home, lambda_away)
    out["match"] = f"{home_team} vs {away_team}"
    return out
