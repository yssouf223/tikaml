# src/selection_engine.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import math


# ============================================================
# CONFIGURATION
# ============================================================

MIN_PROBABILITY_PREMIUM = 0.60
MIN_MARGIN_PREMIUM = 0.10
MIN_CONFIDENCE_PREMIUM = 60.0

MIN_PROBABILITY_GOOD = 0.50
MIN_MARGIN_GOOD = 0.05
MIN_CONFIDENCE_GOOD = 50.0

MIN_PROBABILITY_RISKY = 0.40
MIN_CONFIDENCE_RISKY = 30.0

MAX_DRAW_FOR_WIN_SELECTION = 0.30

# Double chance
DOUBLE_CHANCE_MIN_PROBABILITY = 0.65

# Over / Under
OU_MIN_PROBABILITY = 0.60

# BTTS
BTTS_MIN_PROBABILITY = 0.58


# ============================================================
# OUTILS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_probability(value: Any) -> float:
    """
    Accepte:
      0.65
      65
      "65%"
    et retourne toujours 0.0 - 1.0
    """

    if isinstance(value, str):
        value = value.replace("%", "").strip()

    number = safe_float(value)

    if number > 1:
        number /= 100

    return clamp(number)


def pct(value: float) -> float:
    return round(normalize_probability(value) * 100, 1)


# ============================================================
# DONNEES INTERNES
# ============================================================

@dataclass
class Selection:
    market: str
    pick: str
    probability: float
    confidence: float
    level: str
    risk: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================

class GagneTempsSelectionEngine:

    def __init__(self):
        self.version = "selection-engine-1.0"

    # --------------------------------------------------------
    # NIVEAU GLOBAL
    # --------------------------------------------------------

    def classify(
        self,
        probability: float,
        margin: float,
        confidence: float,
    ) -> Dict[str, Any]:

        probability = normalize_probability(probability)
        margin = safe_float(margin)
        confidence = safe_float(confidence)

        if (
            probability >= MIN_PROBABILITY_PREMIUM
            and margin >= MIN_MARGIN_PREMIUM
            and confidence >= MIN_CONFIDENCE_PREMIUM
        ):
            return {
                "level": "PREMIUM",
                "risk": "FAIBLE",
                "color": "green",
                "score": round(
                    probability * 60
                    + clamp(margin / 0.20) * 20
                    + clamp(confidence / 100) * 20,
                    1,
                ),
            }

        if (
            probability >= MIN_PROBABILITY_GOOD
            and margin >= MIN_MARGIN_GOOD
            and confidence >= MIN_CONFIDENCE_GOOD
        ):
            return {
                "level": "BON PRONOSTIC",
                "risk": "MODÉRÉ",
                "color": "blue",
                "score": round(
                    probability * 60
                    + clamp(margin / 0.20) * 20
                    + clamp(confidence / 100) * 20,
                    1,
                ),
            }

        if (
            probability >= MIN_PROBABILITY_RISKY
            and confidence >= MIN_CONFIDENCE_RISKY
        ):
            return {
                "level": "RISQUÉ",
                "risk": "ÉLEVÉ",
                "color": "orange",
                "score": round(
                    probability * 60
                    + clamp(margin / 0.20) * 20
                    + clamp(confidence / 100) * 20,
                    1,
                ),
            }

        return {
            "level": "À ÉVITER",
            "risk": "TRÈS ÉLEVÉ",
            "color": "red",
            "score": round(probability * 100, 1),
        }

    # --------------------------------------------------------
    # 1X2
    # --------------------------------------------------------

    def analyze_1x2(
        self,
        home_probability: float,
        draw_probability: float,
        away_probability: float,
        home_team: str,
        away_team: str,
    ) -> Dict[str, Any]:

        home = normalize_probability(home_probability)
        draw = normalize_probability(draw_probability)
        away = normalize_probability(away_probability)

        values = {
            "home": home,
            "draw": draw,
            "away": away,
        }

        sorted_values = sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        best_side, best_probability = sorted_values[0]
        second_probability = sorted_values[1][1]

        margin = best_probability - second_probability

        confidence = self.calculate_confidence(
            best_probability,
            margin,
        )

        if best_side == "home":
            pick = home_team
            label = "1"
        elif best_side == "away":
            pick = away_team
            label = "2"
        else:
            pick = "Match nul"
            label = "X"

        classification = self.classify(
            best_probability,
            margin,
            confidence,
        )

        return {
            "market": "1X2",
            "pick": pick,
            "selection": label,
            "probability": pct(best_probability),
            "margin": round(margin, 4),
            "confidence": round(confidence, 1),
            "level": classification["level"],
            "risk": classification["risk"],
            "score": classification["score"],
        }

    # --------------------------------------------------------
    # DOUBLE CHANCE
    # --------------------------------------------------------

    def analyze_double_chance(
        self,
        home_probability: float,
        draw_probability: float,
        away_probability: float,
    ) -> List[Dict[str, Any]]:

        home = normalize_probability(home_probability)
        draw = normalize_probability(draw_probability)
        away = normalize_probability(away_probability)

        candidates = [
            {
                "selection": "1X",
                "probability": home + draw,
            },
            {
                "selection": "X2",
                "probability": draw + away,
            },
            {
                "selection": "12",
                "probability": home + away,
            },
        ]

        candidates.sort(
            key=lambda x: x["probability"],
            reverse=True,
        )

        results = []

        for item in candidates:

            probability = clamp(item["probability"])

            confidence = self.calculate_confidence(
                probability,
                probability - 0.50,
            )

            if probability >= 0.75:
                level = "PREMIUM"
                risk = "FAIBLE"
            elif probability >= DOUBLE_CHANCE_MIN_PROBABILITY:
                level = "BON PRONOSTIC"
                risk = "MODÉRÉ"
            elif probability >= 0.55:
                level = "RISQUÉ"
                risk = "ÉLEVÉ"
            else:
                level = "À ÉVITER"
                risk = "TRÈS ÉLEVÉ"

            results.append({
                "market": "DOUBLE CHANCE",
                "selection": item["selection"],
                "probability": pct(probability),
                "confidence": round(confidence, 1),
                "level": level,
                "risk": risk,
            })

        return results

    # --------------------------------------------------------
    # CALCUL DE CONFIANCE
    # --------------------------------------------------------

    def calculate_confidence(
        self,
        probability: float,
        margin: float,
    ) -> float:

        probability = normalize_probability(probability)

        margin = max(0.0, safe_float(margin))

        probability_score = probability * 70

        margin_score = clamp(
            margin / 0.25
        ) * 30

        return clamp(
            probability_score + margin_score / 100,
            0,
            1,
        ) * 100

    # --------------------------------------------------------
    # POISSON / OVER UNDER
    # --------------------------------------------------------

    def poisson_probability_at_least(
        self,
        expected_goals: float,
        minimum: int,
    ) -> float:

        expected_goals = max(
            0.0,
            safe_float(expected_goals),
        )

        probability_less = 0.0

        for k in range(minimum):
            probability_less += (
                math.exp(-expected_goals)
                * expected_goals ** k
                / math.factorial(k)
            )

        return clamp(1 - probability_less)

    def poisson_probability_at_most(
        self,
        expected_goals: float,
        maximum: int,
    ) -> float:

        expected_goals = max(
            0.0,
            safe_float(expected_goals),
        )

        probability = 0.0

        for k in range(maximum + 1):
            probability += (
                math.exp(-expected_goals)
                * expected_goals ** k
                / math.factorial(k)
            )

        return clamp(probability)

    def analyze_over_under(
        self,
        home_lambda: float,
        away_lambda: float,
    ) -> List[Dict[str, Any]]:

        total_lambda = (
            safe_float(home_lambda)
            + safe_float(away_lambda)
        )

        results = []

        for line in [1.5, 2.5, 3.5]:

            minimum_goals = int(line + 0.5)

            over_probability = self.poisson_probability_at_least(
                total_lambda,
                minimum_goals,
            )

            under_probability = 1 - over_probability

            for label, probability in [
                (
                    f"Over {line}",
                    over_probability,
                ),
                (
                    f"Under {line}",
                    under_probability,
                ),
            ]:

                probability = clamp(probability)

                confidence = probability * 100

                if probability >= 0.70:
                    level = "PREMIUM"
                    risk = "FAIBLE"
                elif probability >= OU_MIN_PROBABILITY:
                    level = "BON PRONOSTIC"
                    risk = "MODÉRÉ"
                elif probability >= 0.50:
                    level = "RISQUÉ"
                    risk = "ÉLEVÉ"
                else:
                    level = "À ÉVITER"
                    risk = "TRÈS ÉLEVÉ"

                results.append({
                    "market": "OVER/UNDER",
                    "selection": label,
                    "probability": pct(probability),
                    "confidence": round(confidence, 1),
                    "level": level,
                    "risk": risk,
                    "expected_goals": round(
                        total_lambda,
                        3,
                    ),
                })

        return sorted(
            results,
            key=lambda x: x["probability"],
            reverse=True,
        )

    # --------------------------------------------------------
    # BTTS
    # --------------------------------------------------------

    def analyze_btts(
        self,
        home_lambda: float,
        away_lambda: float,
    ) -> List[Dict[str, Any]]:

        home_lambda = max(
            0,
            safe_float(home_lambda),
        )

        away_lambda = max(
            0,
            safe_float(away_lambda),
        )

        home_zero = math.exp(-home_lambda)
        away_zero = math.exp(-away_lambda)

        btts_yes = (
            1
            - home_zero
            - away_zero
            + home_zero * away_zero
        )

        btts_no = 1 - btts_yes

        results = []

        for selection, probability in [
            ("BTTS Oui", btts_yes),
            ("BTTS Non", btts_no),
        ]:

            probability = clamp(probability)

            if probability >= 0.70:
                level = "PREMIUM"
                risk = "FAIBLE"
            elif probability >= BTTS_MIN_PROBABILITY:
                level = "BON PRONOSTIC"
                risk = "MODÉRÉ"
            elif probability >= 0.50:
                level = "RISQUÉ"
                risk = "ÉLEVÉ"
            else:
                level = "À ÉVITER"
                risk = "TRÈS ÉLEVÉ"

            results.append({
                "market": "BTTS",
                "selection": selection,
                "probability": pct(probability),
                "confidence": round(
                    probability * 100,
                    1,
                ),
                "level": level,
                "risk": risk,
            })

        return sorted(
            results,
            key=lambda x: x["probability"],
            reverse=True,
        )

    # --------------------------------------------------------
    # SCORE EXACT
    # --------------------------------------------------------

    def analyze_exact_scores(
        self,
        top_scores: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:

        if not top_scores:
            return []

        results = []

        for item in top_scores:

            score = (
                item.get("score")
                or item.get("recommended_score")
                or item.get("selection")
            )

            probability = normalize_probability(
                item.get("probability", 0)
            )

            if not score:
                continue

            if probability >= 0.10:
                level = "RISQUÉ"
                risk = "ÉLEVÉ"
            elif probability >= 0.07:
                level = "À ÉVITER"
                risk = "TRÈS ÉLEVÉ"
            else:
                level = "À ÉVITER"
                risk = "TRÈS ÉLEVÉ"

            results.append({
                "market": "SCORE EXACT",
                "selection": score,
                "probability": pct(probability),
                "confidence": round(
                    probability * 100,
                    1,
                ),
                "level": level,
                "risk": risk,
            })

        return results

    # --------------------------------------------------------
    # CORNERS
    # --------------------------------------------------------

    def analyze_corners(
        self,
        home_corners: Optional[float],
        away_corners: Optional[float],
    ) -> List[Dict[str, Any]]:

        if home_corners is None or away_corners is None:
            return []

        total = (
            safe_float(home_corners)
            + safe_float(away_corners)
        )

        results = []

        # Estimation prudente autour de la moyenne de Poisson.
        for line in [7.5, 8.5, 9.5, 10.5]:

            over = self.poisson_probability_at_least(
                total,
                int(line + 0.5),
            )

            under = 1 - over

            for selection, probability in [
                (f"Over {line} corners", over),
                (f"Under {line} corners", under),
            ]:

                if probability >= 0.70:
                    level = "PREMIUM"
                    risk = "FAIBLE"
                elif probability >= 0.60:
                    level = "BON PRONOSTIC"
                    risk = "MODÉRÉ"
                elif probability >= 0.50:
                    level = "RISQUÉ"
                    risk = "ÉLEVÉ"
                else:
                    level = "À ÉVITER"
                    risk = "TRÈS ÉLEVÉ"

                results.append({
                    "market": "CORNERS",
                    "selection": selection,
                    "probability": pct(probability),
                    "confidence": round(
                        probability * 100,
                        1,
                    ),
                    "level": level,
                    "risk": risk,
                    "expected_corners": round(
                        total,
                        2,
                    ),
                })

        return sorted(
            results,
            key=lambda x: x["probability"],
            reverse=True,
        )

    # --------------------------------------------------------
    # CARTONS
    # --------------------------------------------------------

    def analyze_cards(
        self,
        home_cards: Optional[float],
        away_cards: Optional[float],
    ) -> List[Dict[str, Any]]:

        if home_cards is None or away_cards is None:
            return []

        total = (
            safe_float(home_cards)
            + safe_float(away_cards)
        )

        results = []

        for line in [3.5, 4.5, 5.5]:

            over = self.poisson_probability_at_least(
                total,
                int(line + 0.5),
            )

            under = 1 - over

            for selection, probability in [
                (f"Over {line} cartons", over),
                (f"Under {line} cartons", under),
            ]:

                if probability >= 0.70:
                    level = "PREMIUM"
                    risk = "FAIBLE"
                elif probability >= 0.60:
                    level = "BON PRONOSTIC"
                    risk = "MODÉRÉ"
                elif probability >= 0.50:
                    level = "RISQUÉ"
                    risk = "ÉLEVÉ"
                else:
                    level = "À ÉVITER"
                    risk = "TRÈS ÉLEVÉ"

                results.append({
                    "market": "CARTONS",
                    "selection": selection,
                    "probability": pct(probability),
                    "confidence": round(
                        probability * 100,
                        1,
                    ),
                    "level": level,
                    "risk": risk,
                    "expected_cards": round(
                        total,
                        2,
                    ),
                })

        return sorted(
            results,
            key=lambda x: x["probability"],
            reverse=True,
        )

    # --------------------------------------------------------
    # ANALYSE COMPLETE D'UN MATCH
    # --------------------------------------------------------

    def analyze_match(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:

        home_team = (
            prediction.get("home_team")
            or prediction.get("home")
            or ""
        )

        away_team = (
            prediction.get("away_team")
            or prediction.get("away")
            or ""
        )

        probabilities = prediction.get(
            "probabilities",
            {},
        )

        home_probability = normalize_probability(
            probabilities.get(
                "home",
                prediction.get("home_probability", 0),
            )
        )

        draw_probability = normalize_probability(
            probabilities.get(
                "draw",
                prediction.get("draw_probability", 0),
            )
        )

        away_probability = normalize_probability(
            probabilities.get(
                "away",
                prediction.get("away_probability", 0),
            )
        )

        # ----------------------------------------------------
        # 1X2
        # ----------------------------------------------------

        one_x_two = self.analyze_1x2(
            home_probability,
            draw_probability,
            away_probability,
            home_team,
            away_team,
        )

        # ----------------------------------------------------
        # DOUBLE CHANCE
        # ----------------------------------------------------

        double_chance = self.analyze_double_chance(
            home_probability,
            draw_probability,
            away_probability,
        )

        # ----------------------------------------------------
        # LAMBDA
        # ----------------------------------------------------

        lambdas = prediction.get(
            "lambdas",
            {},
        )

        home_lambda = safe_float(
            lambdas.get(
                "home",
                prediction.get("lambda_home", 0),
            )
        )

        away_lambda = safe_float(
            lambdas.get(
                "away",
                prediction.get("lambda_away", 0),
            )
        )

        # ----------------------------------------------------
        # OVER UNDER
        # ----------------------------------------------------

        over_under = self.analyze_over_under(
            home_lambda,
            away_lambda,
        )

        # ----------------------------------------------------
        # BTTS
        # ----------------------------------------------------

        btts = self.analyze_btts(
            home_lambda,
            away_lambda,
        )

        # ----------------------------------------------------
        # SCORE EXACT
        # ----------------------------------------------------

        top_scores = prediction.get(
            "top_scores",
            prediction.get(
                "score_predictions",
                [],
            ),
        )

        exact_scores = self.analyze_exact_scores(
            top_scores
        )

        # ----------------------------------------------------
        # CORNERS
        # ----------------------------------------------------

        corners = prediction.get(
            "corners",
            {},
        )

        corners_results = self.analyze_corners(
            corners.get("home"),
            corners.get("away"),
        )

        # ----------------------------------------------------
        # CARTONS
        # ----------------------------------------------------

        cards = prediction.get(
            "cards",
            prediction.get(
                "yellows",
                {},
            ),
        )

        cards_results = self.analyze_cards(
            cards.get("home"),
            cards.get("away"),
        )

        # ----------------------------------------------------
        # MEILLEURE SELECTION
        # ----------------------------------------------------

        candidates = []

        candidates.append(one_x_two)

        candidates.extend(
            double_chance
        )

        candidates.extend(
            over_under
        )

        candidates.extend(
            btts
        )

        candidates.extend(
            corners_results
        )

        candidates.extend(
            cards_results
        )

        valid_candidates = [
            x for x in candidates
            if x.get("level") != "À ÉVITER"
        ]

        valid_candidates.sort(
            key=lambda x: (
                x.get("score", 0),
                x.get("confidence", 0),
                x.get("probability", 0),
            ),
            reverse=True,
        )

        best_selection = (
            valid_candidates[0]
            if valid_candidates
            else one_x_two
        )

        # ----------------------------------------------------
        # NIVEAU GLOBAL DU MATCH
        # ----------------------------------------------------

        global_probability = (
            normalize_probability(
                best_selection.get(
                    "probability",
                    0,
                )
            )
        )

        global_confidence = safe_float(
            best_selection.get(
                "confidence",
                0,
            )
        )

        if (
            best_selection.get("level")
            == "PREMIUM"
        ):
            global_level = "PREMIUM"
            global_risk = "FAIBLE"

        elif (
            best_selection.get("level")
            == "BON PRONOSTIC"
        ):
            global_level = "BON PRONOSTIC"
            global_risk = "MODÉRÉ"

        elif (
            best_selection.get("level")
            == "RISQUÉ"
        ):
            global_level = "RISQUÉ"
            global_risk = "ÉLEVÉ"

        else:
            global_level = "À ÉVITER"
            global_risk = "TRÈS ÉLEVÉ"

        # ----------------------------------------------------
        # SORTIE
        # ----------------------------------------------------

        return {
            "engine": self.version,

            "match": {
                "home_team": home_team,
                "away_team": away_team,
            },

            "global_selection": {
                "market": best_selection.get(
                    "market"
                ),
                "selection": best_selection.get(
                    "selection",
                    best_selection.get(
                        "pick"
                    ),
                ),
                "probability": best_selection.get(
                    "probability"
                ),
                "confidence": best_selection.get(
                    "confidence"
                ),
                "level": global_level,
                "risk": global_risk,
            },

            "markets": {
                "1x2": one_x_two,
                "double_chance": double_chance,
                "over_under": over_under,
                "btts": btts,
                "exact_scores": exact_scores,
                "corners": corners_results,
                "cards": cards_results,
            },

            "model_context": {
                "home_probability": pct(
                    home_probability
                ),
                "draw_probability": pct(
                    draw_probability
                ),
                "away_probability": pct(
                    away_probability
                ),
                "home_lambda": round(
                    home_lambda,
                    4,
                ),
                "away_lambda": round(
                    away_lambda,
                    4,
                ),
            },
        }


# ============================================================
# INSTANCE GLOBALE
# ============================================================

selection_engine = GagneTempsSelectionEngine()


# ============================================================
# FONCTION SIMPLE POUR LE SERVEUR
# ============================================================

def select_prediction(
    prediction: Dict[str, Any],
) -> Dict[str, Any]:

    return selection_engine.analyze_match(
        prediction
    )
