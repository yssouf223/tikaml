# TikaML Core — Model Documentation

## 1. Overview

TikaML Core is a machine learning-based football match prediction system built on the LightGBM gradient boosting framework with Poisson regression. The system consists of three independent models that predict goals, corners, and yellow cards, covering both pre-match predictions and live (in-play) real-time updates.

Trained on over 17,000 matches from Europe's top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), the goal model achieves a **Ranked Probability Score (RPS) of 0.1968** under strict forward-chain temporal validation.

---

## 2. Three-Model Architecture

TikaML Core runs three parallel, independent Poisson regression models. Each model consists of two separate regressors that predict the expected rates (λ) for the home and away teams independently.

### 2.1 Goals Model (84 Features)

The goals model is the core of the system. It predicts the expected goals for home and away teams (λ_home and λ_away) and constructs a full **7×7 score probability matrix**.

The matrix undergoes two post-processing corrections:

- **Dixon-Coles Low-Score Correction** (ρ = -0.108): Adjusts the probabilities for the four low-score cells (0-0, 1-0, 0-1, 1-1) to correct for the independence assumption bias inherent in the Poisson model at low scores. This is a standard correction in football prediction, originating from Dixon & Coles (1997).

- **Temperature Scaling** (T = 0.90): Sharpens the probability distribution to correct the model's conservative bias at high confidence levels. When the model assigns high probabilities (e.g., 75%+), actual win rates tend to be even higher; temperature scaling partially closes this gap.

All prediction outputs are derived from the 7×7 matrix:

| Output | Description |
|--------|-------------|
| 1x2 Probabilities | Full probability distribution over home win, draw, and away win |
| Recommended Score | The single most probable scoreline from the matrix |
| Over/Under | O/U probabilities for 1.5, 2.5, and 3.5 goal lines |
| Score Matrix | Complete 7×7 score probability distribution (with Dixon-Coles correction) |

### 2.2 Corners Model (89 Features)

Extends the 84 base features with 5 corner-specific features:

- Team corner averages over the last 10 matches (home/away)
- Team conceded corner averages over the last 10 matches (home/away)
- Match referee's historical corner average

Outputs over/under probabilities for: O/U 8.5, 9.5, 10.5, 11.5.

### 2.3 Yellow Cards Model (91 Features)

Extends the 84 base features with 7 card/foul-specific features:

- Team yellow card averages over the last 10 matches (home/away)
- Team conceded yellow card averages over the last 10 matches (home/away)
- Team foul averages over the last 10 matches (home/away)
- Match referee's historical yellow card average

Outputs over/under probabilities for: O/U 2.5, 3.5, 4.5, 5.5.

---

## 3. Feature Engineering

### 3.1 Feature Categories

The model uses **84 base features** across 12 categories. All features are computable from pre-match information only — there is no data leakage.

| Category | Features | Count | Description |
|----------|----------|-------|-------------|
| Rolling xG | Expected goals and conceded xG, exponentially weighted | 8 | Core attacking/defensive capability |
| Rolling Tactical | Shots, PPDA, progressive passes/carries, box touches, etc. | 16 | Team tactical profile |
| Form | Goals scored and conceded rolling averages | 4 | Recent scoring ability |
| Venue-Specific | Home-only and away-only xG, shots, goals | 8 | Home/away performance split |
| Derived | xG overperformance, shot accuracy, clean sheet percentage | 6 | Composite metrics |
| Head-to-Head | Historical win rate, goal difference, match count | 3 | Direct rivalry record |
| Relative Strength | xG, shots vs league average ratios | 6 | Strength within competition |
| Momentum | Short-term vs long-term form delta | 2 | Rising/declining trajectory |
| Draw Tendency | Team/league draw rates, xG gap, defensive strength | 5 | Draw-proneness indicators |
| Match Context | Rest days, midweek flag, season stage, points/position gap | 8 | Schedule and standings |
| Match Importance | Relegation battle, title race, safety margin, composite score | 5 | Competitive pressure |
| Lineup Rotation | Changes, stability, formation shifts, rotation rate | 8 | Squad management |
| Market Odds | Bookmaker implied probabilities (home/draw/away) | 3 | Market consensus signal |
| **Base Total** | | **84** | |
| Corner-Specific | Corner rolling, conceded, referee corners | +5 | Corners model only |
| Yellow-Specific | Yellow rolling, conceded, fouls rolling, referee yellows | +7 | Yellows model only |

### 3.2 Rolling Feature Computation

All rolling features use exponentially weighted moving averages (EWM) over a 10-match window with a half-life of 5. This means the most recent match carries approximately 4x the weight of the 10th most recent match, balancing recency with stability.

Venue-split features further separate rolling statistics into home-only and away-only groups to capture performance differences across environments.

### 3.3 Missing Value Handling

LightGBM natively supports missing values. The model handles NaN inputs without imputation during both training and inference. Odds features (~7% missing) and league table features (unavailable for cross-competition predictions) are passed directly as NaN.

---

## 4. Training Data

### 4.1 Data Sources and Scale

| Dimension | Detail |
|-----------|--------|
| Leagues | Premier League (EPL), La Liga, Serie A, Bundesliga, Ligue 1 |
| Time Span | 12 seasons (2014–2026) |
| Total Matches | 21,121 (17,469 from 2016+ used for training) |
| Feature Dimensions | 245 columns |
| Odds Coverage | ~93% for 2018+ seasons (EPL/La Liga/Serie A ≈ 100%, Bundesliga 84%, Ligue 1 76%) |
| Corner Data Coverage | 17,207 matches (98.5%) |
| Yellow Card Data Coverage | 14,902 matches (85.3%) |

### 4.2 Data Sources

- **Match Event Data**: Opta (Stats Perform) official API — detailed event-level statistics including shots, passes, possession, xG, and more
- **Betting Odds**: Football-Data.co.uk historical odds from major bookmakers including Pinnacle and Bet365
- **Referee Data**: Extracted from Opta match data, computing per-referee historical officiating statistics

---

## 5. Validation Methodology and Performance

### 5.1 Forward-Chain Temporal Validation

The model is evaluated using strict **forward-chain validation** (expanding window):

- For test season S, the model is trained exclusively on all seasons before S
- The last training season serves as the validation set for early stopping
- **No future data leakage**: the model never sees any future match information during training

### 5.2 Goals Model Performance

| Test Season | Matches | RPS ↓ | Accuracy | High-Confidence (≥60%) Accuracy |
|-------------|---------|-------|----------|--------------------------------|
| 2022-2023 | 1,827 | 0.2021 | 53.1% | 68.0% |
| 2023-2024 | 1,752 | 0.1908 | 54.2% | 71.1% |
| 2024-2025 | 1,750 | 0.1974 | 53.5% | 67.5% |
| 2025-2026 | 1,290 | 0.1990 | 52.4% | 70.3% |
| **4-Season Average** | | **0.1968** | **53.3%** | **69.2%** |

**RPS (Ranked Probability Score)** is the primary evaluation metric. It measures the quality of the full probability distribution rather than just the top prediction. Lower is better.

### 5.3 Confidence and Accuracy

The maximum of the three 1x2 probabilities serves as the confidence indicator. Accuracy varies significantly across confidence levels:

| Confidence Range | Interpretation | Approximate Accuracy |
|-----------------|----------------|---------------------|
| < 45% | Difficult to predict, evenly matched teams | ~40% |
| 45–55% | Moderate certainty | ~52% |
| 55–65% | Fairly certain | ~60% |
| 65–75% | High certainty | ~72% |
| 75–90% | Very high certainty, typically lopsided matchups | ~87% |

### 5.4 Probability Calibration

| Predicted P(Home Win) | Actual Home Win Rate | Bias |
|-----------------------|---------------------|------|
| 35–45% | 40.5% | -0.6% |
| 45–55% | 52.0% | -2.2% |
| 55–65% | 60.2% | -0.3% |
| 65–75% | 72.1% | -2.4% |
| 75–90% | 87.0% | -7.9% |

The model is well-calibrated overall — predicted probabilities closely match observed frequencies. A slight conservative bias exists at high confidence levels, partially corrected by temperature scaling at T = 0.90.

### 5.5 Corners and Yellow Cards Performance

| Model | Metric | Performance | vs Mean Baseline | Improvement |
|-------|--------|-------------|-----------------|-------------|
| Corners | MAE (Home) | 2.177 | 2.300 | +5.3% |
| Corners | MAE (Away) | 1.912 | 1.968 | +2.9% |
| Yellow Cards | MAE (Home) | 0.926 | 0.934 | +0.8% |
| Yellow Cards | MAE (Away) | 0.995 | 1.025 | +2.9% |

Corner over/under accuracy: O8.5 62.4%, O9.5 51.6%, O10.5 60.6%, O11.5 71.9% — well-calibrated across all lines.

### 5.6 RPS Loss Decomposition

| Actual Outcome | Share | Average RPS | Loss Contribution |
|---------------|-------|-------------|-------------------|
| Home Win | 44% | 0.175 | 39% |
| Draw | 25% | 0.159 | 20% |
| Away Win | 31% | 0.267 | 41% |

Away upsets are the largest source of prediction error — when the model assigns high home-win probability but the away team wins, the per-match RPS penalty is severe.

---

## 6. Live Prediction Engine

### 6.1 How It Works

The live engine takes the pre-match λ_home and λ_away as priors and dynamically adjusts predictions through Bayesian updating as the match progresses.

Core update mechanisms:

| Mechanism | Parameter | Description |
|-----------|-----------|-------------|
| Non-linear time decay | r^0.82 | Remaining time raised to the 0.82 power, reflecting the tendency for goals to cluster in late periods |
| 6-bucket score momentum | See table below | Adjusts both teams' attacking intensity based on the current score differential |
| Red card factors | 0.61× / 1.46× | A red card dramatically reduces the team's attack (×0.61) and boosts the opponent (×1.46) |
| Dixon-Coles decay | ρ decays linearly | Low-score correction diminishes linearly as the match progresses |

### 6.2 Score Momentum Parameters

| Score Differential | Attack Adjustment | Opponent Attack Adjustment | Interpretation |
|-------------------|-------------------|---------------------------|----------------|
| Trailing by 3+ | ×1.40 | ×0.85 | Desperate all-out attack; opponent drops deep |
| Trailing by 2 | ×1.13 | ×1.09 | Both sides increase attacking intensity |
| Trailing by 1 | ×1.00 | ×0.87 | Maintain attack; opponent contracts |
| Leading by 1 | ×0.88 | ×1.03 | Slight contraction |
| Leading by 2 | ×0.87 | ×1.06 | Game management mode |
| Leading by 3+ | ×0.85 | ×1.01 | Opponent effectively gives up the chase |

### 6.3 Live Outputs

In live state, the goals model produces additional outputs beyond the standard pre-match fields:

- **Remaining λ**: Expected goals for each team in the remaining match time
- **Next Goal Probability**: Probability of home goal / away goal / no more goals
- **Final Score Matrix**: 7×7 probability distribution for the final score, derived from the remaining-time matrix
- **Recommended Score**: The most probable final scoreline

Corners and yellow cards use simple linear time decay in live state (no score momentum). Remaining λ = pre-match λ × (90 - current minute) / 90, with over/under probabilities recalculated from current count + remaining λ.

### 6.4 Live Accuracy

Validated against Opta's official live predictions across 150 EPL and La Liga matches (MAE = 0.030):

| Period | Opta Accuracy | TikaML Accuracy | Pre-Match Model (Static) |
|--------|--------------|-----------------|--------------------------|
| 0–15' | 64% | 64% | ~53% |
| 60–75' | 70% | 67% | ~53% |
| 75–90' | 86% | 83% | ~53% |

The live engine's accuracy improves significantly as the match progresses. It matches Opta in early stages and trails by 2-3% in the 60-75' window, primarily because Opta incorporates real-time in-game statistics (shots, possession, xG) that are unavailable to our model.

---

## 7. Prediction Output Details

### 7.1 Pre-Match Prediction Output

```
Goals:
  1x2 Probabilities  — Home 47.5% / Draw 27.7% / Away 24.8%
  Recommended Score   — 1-1 (13.2%)
  Over/Under          — O1.5: 75.7%  O2.5: 49.7%  O3.5: 27.7%
  Score Matrix        — 7×7 probability distribution (with Dixon-Coles correction)

Corners:
  Predicted Total     — 10.2 (Home 5.7, Away 4.5)
  Over/Under          — O8.5: 68.4%  O9.5: 56.1%  O10.5: 43.6%  O11.5: 32.1%

Yellow Cards:
  Predicted Total     — 4.5 (Home 2.1, Away 2.4)
  Over/Under          — O2.5: 82.1%  O3.5: 65.0%  O4.5: 45.9%  O5.5: 28.9%
```

### 7.2 Live Prediction Output (Example: 55', Score 1-0)

```
Goals:
  1x2 Probabilities   — Home 75.9% / Draw 18.6% / Away 5.5%
  Recommended Score    — 1-0 (31.9%)
  Remaining λ          — Home 0.64, Away 0.51
  Next Goal            — Home 37.8% / Away 30.3% / None 31.9%
  Over/Under           — O2.5: 32.5%  O3.5: 11.1%

Corners (Current: 5-3):
  Remaining λ          — Home 2.2, Away 1.7
  Over/Under           — O8.5: 98.1%  O9.5: 90.5%  O10.5: 75.4%  O11.5: 55.7%

Yellow Cards (Current: 1-1):
  Remaining λ          — Home 0.8, Away 0.9
  Over/Under           — O2.5: 82.3%  O3.5: 51.7%  O4.5: 25.1%  O5.5: 9.8%
```

### 7.3 Understanding Confidence

The confidence level for 1x2 predictions equals the maximum of the three probabilities. Higher confidence correlates with higher accuracy:

- **Low confidence** (< 45%): Evenly matched teams or weak data signal; high prediction uncertainty
- **Medium confidence** (45–60%): A lean toward one outcome, but others remain plausible
- **High confidence** (≥ 60%): The model is fairly certain; historical accuracy approximately 69%

For score predictions, confidence equals the recommended score's probability. Due to the inherent dispersion of football scorelines, even the most likely score typically has only a 10-15% probability — but calibration is reliable across probability levels.

---

## 8. Research and Development

### 8.1 Model Evolution

The system underwent systematic methodology exploration, evolving from traditional statistical models to the current architecture:

| Stage | RPS | Key Change |
|-------|-----|------------|
| Dixon-Coles Baseline | 0.2060 | Classical Poisson model with MLE |
| Elo Ratings → Poisson | ~0.205 | Elo-based λ estimation |
| LightGBM Poisson (68 features) | 0.1997 | Optuna hyperparameter optimization |
| + Venue-specific & derived features | 0.1993 | 8 + 6 new features |
| + Market odds implied probabilities | 0.1979 | Largest single improvement |
| + Temperature scaling | 0.1976 | Conservative bias correction |
| + Match importance & lineup rotation | 0.1973 | 84 features |
| + Odds coverage correction | **0.1968** | Final model |

**Total improvement**: 0.2060 → 0.1968, an RPS reduction of 0.0092.

### 8.2 Methods That Did Not Improve Results

The following approaches were rigorously tested and ultimately not adopted:

| Method | Outcome | Reason |
|--------|---------|--------|
| Model ensembling (stacking/bagging) | No gain | When the primary model has rich features, secondary models contribute virtually no orthogonal information |
| Neural networks (team embeddings) | No gain | Insufficient data (~17K samples) for effective embedding learning |
| Bivariate Poisson (correlation parameter λ₃) | No gain | λ₃ highly unstable across seasons |
| Draw specialist classifier | No gain | P(draw) for actual draws ≈ P(draw) for non-draws (0.270 vs 0.269) |
| Player-level XI features (27 variants) | No gain | Team rolling stats already encode player contributions; odds already price lineups |
| UCL combined training | No gain | Only 433 matchable UCL games; zero-shot transfer already achieves RPS = 0.1946 on group stage |
| Opta pre-match win probability | No gain | Highly correlated with bookmaker odds (r² ≈ 0.98), redundant information |

### 8.3 Key Insight

**Single-model feature enrichment consistently outperformed multi-model ensembling.** When the primary model has access to rich features including market odds, any secondary model's marginal contribution is negligible. This differs from common Kaggle competition experience because the signal in football prediction is inherently weak — stacking multiple weak signals yields far less gain than adding a single strong feature.

---

## 9. Cross-Competition Predictions

### 9.1 UEFA Champions League

TikaML Core supports Champions League predictions via zero-shot transfer — using the top-5-league-trained model directly without additional training.

- Each team's features are derived from their most recent domestic league match rolling statistics
- League table features (points, position, etc.) are passed as NaN (handled natively by LightGBM)
- Limited to matches where both teams have top-5-league data (approximately 31% of all UCL matches)

Validation results: UCL group stage RPS = 0.1946 (better than the domestic league baseline of 0.1968), knockout stage RPS = 0.2189 (smaller sample, higher uncertainty).

### 9.2 Why Not Combined Training

Experiments showed that merging UCL data into the training set did not improve UCL predictions (+0.0002 RPS) and slightly harmed domestic league predictions (+0.0004 RPS). With only 433 matchable UCL games, the signal was too weak and introduced noise.

---

## 10. Known Limitations

### 10.1 Structural Limitation on Draw Prediction

The model never outputs "Draw" as the most likely 1x2 outcome. This is a **mathematical property of the Poisson distribution**, not a model deficiency: when λ ≥ 0.89 (a condition met by all real football matches), the sum of draw probabilities (0-0 + 1-1 + 2-2 + ...) is always less than the sum of either home win or away win probabilities.

However, the model's aggregate P(Draw) is well-calibrated (predicted 25.6% vs actual 25.3%), and individual draw scorelines (especially 1-1) frequently appear as the single most probable score in the 7×7 matrix.

### 10.2 Upset Sensitivity

Away upsets (strong home team losing) contribute 41% of total RPS loss despite away wins accounting for only 31% of outcomes. Upsets are primarily driven by factors unobservable before kickoff — injuries, match-day motivation, tactical surprises. Market odds features partially mitigate this but cannot fully resolve it.

### 10.3 Gap to Bookmaker Performance

| | RPS |
|--|-----|
| TikaML | 0.1968 |
| Bookmaker Consensus | ~0.185 |
| Gap | +0.012 |

The gap is attributed to:

- **Real-time information** (~60%): Injury updates, confirmed lineups, transfer activity, player fitness. Bookmakers employ dedicated teams to track these in real time.
- **Market wisdom** (~25%): The collective judgment of thousands of bettors creates a highly efficient information aggregation mechanism.
- **Model structure** (~15%): Poisson distributional constraints on low-score matches and the inherent unpredictability of draws.

---

## 11. References

- Dixon, M. J., & Coles, S. G. (1997). *Modelling association football scores and inefficiencies in the football betting market.* Journal of the Royal Statistical Society: Series C, 46(2), 265-280.
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022.
- Constantinou, A. C., & Fenton, N. E. (2012). *Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models.* Journal of Quantitative Analysis in Sports, 8(1).
- Shwartz-Ziv, R., & Armon, A. (2022). *Tabular data: Deep learning is not all you need.* Information Fusion, 81, 84-90.
