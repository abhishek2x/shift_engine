# ShiftEngine \- Technical Mathematics & Architecture Guide

This document breaks down the mathematical foundation of ShiftEngine, explicitly mapping theoretical concepts to the actual variables, functions, and files inside the Python codebase.

---

## 1. The Core Equation: Bayes' Theorem

At its heart, ShiftEngine executes recursive Bayes' Theorem.

$$ P(State_{t} | Return_{t}) = \frac{P(Return_{t} | State_{t}) \cdot P(State_{t-1})}{P(Return_{t})} $$

### Definitions

| Term | Symbol | Meaning |
|------|--------|---------|
| **Prior** | $P(State_{t-1})$ | Your belief *before* seeing the new market data. |
| **Likelihood** | $P(Return_{t} \| State_{t})$ | The probability that a normal distribution for a given state generated the observed return. |
| **Marginal Likelihood** | $P(Return_{t})$ | **The denominator.** The total probability of seeing the return across *all* states. Calculated via the Law of Total Probability. |
| **Posterior** | $P(State_{t} \| Return_t)$ | Your updated belief. Becomes tomorrow's Prior recursively. |

---

## 2. The Computational Danger: Floating Point Underflow

In recursive Bayesian updates, you continuously multiply extremely small probability decimals (e.g., $0.05 \times 0.01 \times 0.20 \dots$).

Very quickly, these chained multiplications result in a number smaller than a CPU's floating-point precision limit (typically $10^{-320}$). When this ceiling is breached, the program undergoes **Underflow** and rounds the probability to exactly `0.0`. Once a probability hits zero, it can never multiply back up. The mathematical engine is permanently broken.

**The Solution:** Operate entirely in **Log-Space**. The natural logarithm transforms multiplication into addition: $\ln(A \times B) = \ln(A) + \ln(B)$. Instead of multiplying shrinking decimals, we simply add negative numbers, which Python handles easily.

---

## 3. Regime Configuration (Dual-Distribution Modeling)

Before the engine runs, you must define two normal distributions — one for each market state:

| Regime | Mean ($\mu$) | Std Dev ($\sigma$) | Interpretation |
|--------|-------------|---------------------|----------------|
| **Bull** (Low Vol) | $+0.0005$ | $0.01$ | Market trends up slowly with tight, calm price action. |
| **Bear** (High Vol) | $-0.0010$ | $0.025$ | Market swings wildly with a downward drift. |

These parameters are what allow the **same function** (`log_normal_pdf`) to produce **different outputs** for Bull vs Bear — because the `mean` and `std_dev` arguments change between the two calls.

---

## 4. The Implementation Flow

**Entry point:** `compute_posterior_log_space()` in **`src/math_utils.py` (line 30)**

When a new market return arrives, this function orchestrates the entire Bayes update by calling the other functions internally.

```
New Market Return (x)
        │
        ▼
┌─────────────────────────────────────────────┐
│  compute_posterior_log_space()               │
│  File: src/math_utils.py, line 30           │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ STEP 1: Calculate Likelihoods       │    │
│  │ Calls: log_normal_pdf() TWICE       │    │
│  │ File: src/math_utils.py, line 20    │    │
│  │                                     │    │
│  │ Call 1 (Bull): log_normal_pdf(      │    │
│  │   x, bull_mean, bull_std_dev)       │    │
│  │   → log_likelihood_bull             │    │
│  │                                     │    │
│  │ Call 2 (Bear): log_normal_pdf(      │    │
│  │   x, bear_mean, bear_std_dev)       │    │
│  │   → log_likelihood_bear             │    │
│  └─────────────────────────────────────┘    │
│                    │                        │
│                    ▼                        │
│  ┌─────────────────────────────────────┐    │
│  │ STEP 2: Numerators (Prior × Likeli) │    │
│  │ In log-space: addition replaces     │    │
│  │ multiplication.                     │    │
│  │                                     │    │
│  │ unnorm_bull = log_likelihoods[0]    │    │
│  │             + prior_log_probs[0]    │    │
│  │                          (line 45)  │    │
│  │ unnorm_bear = log_likelihoods[1]    │    │
│  │             + prior_log_probs[1]    │    │
│  │                          (line 46)  │    │
│  └─────────────────────────────────────┘    │
│                    │                        │
│                    ▼                        │
│  ┌─────────────────────────────────────┐    │
│  │ STEP 3: Denominator (LSE Trick)     │    │
│  │ Calls: log_sum_exp()                │    │
│  │ File: src/math_utils.py, line 4     │    │
│  │                                     │    │
│  │ marginal_log_prob = log_sum_exp(    │    │
│  │   [unnorm_bull, unnorm_bear])       │    │
│  │                          (line 50)  │    │
│  └─────────────────────────────────────┘    │
│                    │                        │
│                    ▼                        │
│  ┌─────────────────────────────────────┐    │
│  │ STEP 4: Normalize (Posteriors)      │    │
│  │ In log-space: subtraction replaces  │    │
│  │ division.                           │    │
│  │                                     │    │
│  │ posterior_bull = unnorm_bull         │    │
│  │               - marginal_log_prob   │    │
│  │                          (line 53)  │    │
│  │ posterior_bear = unnorm_bear         │    │
│  │               - marginal_log_prob   │    │
│  │                          (line 54)  │    │
│  └─────────────────────────────────────┘    │
│                    │                        │
│                    ▼                        │
│  Return (posterior_bull, posterior_bear)     │
│  These become tomorrow's prior_log_probs!   │
└─────────────────────────────────────────────┘
```

---

## 5. Detailed Function Reference

### 5.1 `log_normal_pdf(x, mean, std_dev)` → `float`
> **File:** `src/math_utils.py`, **line 20**
> **Concept:** Calculates the Likelihood term of Bayes' Theorem in log-space.

**Formula:**
$$ \ln P(x|\mu, \sigma) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2} $$

**Why we call it twice with different parameters:**
The function is regime-agnostic. It only knows math. The *caller* decides the regime by passing different `mean` and `std_dev` values:
*   **Bull call:** `log_normal_pdf(-0.03, 0.0005, 0.01)` → returns a very negative number (unlikely).
*   **Bear call:** `log_normal_pdf(-0.03, -0.0010, 0.025)` → returns a less negative number (more likely).

---

### 5.2 `log_sum_exp(log_probs)` → `float`
> **File:** `src/math_utils.py`, **line 4**
> **Concept:** Safely computes the Marginal Likelihood (Denominator) of Bayes' Theorem in log-space using the **Log-Sum-Exp (LSE) Trick**.

**The Problem It Solves:**
We need to calculate $\ln(e^A + e^B)$. But directly computing $e^A$ when $A$ is deeply negative (e.g., $-100$) triggers Floating Point Underflow.

**The Algebraic Trick:**
Let $c = \max(A, B)$.
1.  Start with: $\ln(e^A + e^B)$
2.  Factor out $e^c$: $\ln \left[ e^c \cdot (e^{A-c} + e^{B-c}) \right]$
3.  Split the log: $\ln(e^c) + \ln(e^{A-c} + e^{B-c})$
4.  Simplify: $c + \ln(e^{A-c} + e^{B-c})$

**Code-to-Math Mapping:**

| Code Variable | Math Symbol | Description |
|---------------|-------------|-------------|
| `log_probs` | $[A, B]$ | The array of unnormalized log-probabilities |
| `max_log` (line 9) | $c$ | $\max(A, B)$ — the largest value, used as the anchor |
| `lp - max_log` (line 16) | $A - c$ or $B - c$ | Subtracting the anchor before exponentiating (prevents crash) |
| `sum_res` (line 14) | $e^{A-c} + e^{B-c}$ | The safe sum of shifted exponents |
| Return value (line 18) | $c + \ln(\text{sum\_res})$ | The final, numerically stable result |

**Edge Case (line 11):** If all log-probabilities are $-\infty$, the function returns $-\infty$ instead of crashing on `math.log(0)`.

---

### 5.3 `compute_posterior_log_space(prior_log_probs, log_likelihoods)` → `Tuple[float, float]`
> **File:** `src/math_utils.py`, **line 30**
> **Concept:** The main orchestrator. Executes a single recursive Bayes update step in log-space by calling the above two functions.

**Code-to-Math Mapping:**

| Code Variable | Math Symbol | Bayes Role |
|---------------|-------------|------------|
| `prior_log_probs[0]` | $\ln P(Bull_{t-1})$ | Prior belief of Bull |
| `prior_log_probs[1]` | $\ln P(Bear_{t-1})$ | Prior belief of Bear |
| `log_likelihoods[0]` | $\ln P(Return \| Bull)$ | Likelihood from Bull distribution |
| `log_likelihoods[1]` | $\ln P(Return \| Bear)$ | Likelihood from Bear distribution |
| `unnorm_bull` (line 45) | $\ln P(Return \| Bull) + \ln P(Bull)$ | Numerator for Bull |
| `unnorm_bear` (line 46) | $\ln P(Return \| Bear) + \ln P(Bear)$ | Numerator for Bear |
| `marginal_log_prob` (line 50) | $\ln P(Return)$ | Denominator (via `log_sum_exp`) |
| `posterior_bull` (line 53) | $\ln P(Bull \| Return)$ | Final updated Bull belief |
| `posterior_bear` (line 54) | $\ln P(Bear \| Return)$ | Final updated Bear belief |

---

## 6. Worked Numerical Example

**Setup:** Start with a neutral 50/50 belief. A market return of $x = -0.03$ (-3%) arrives.
*   `prior_log_probs = (ln(0.50), ln(0.50))` = `(-0.69, -0.69)`

### Step 1: Likelihoods via `log_normal_pdf()`
| Call | Arguments | Output | Interpretation |
|------|-----------|--------|----------------|
| Bull | `log_normal_pdf(-0.03, 0.0005, 0.01)` | `-5.11` | A -3% drop is wildly unlikely in a calm Bull market |
| Bear | `log_normal_pdf(-0.03, -0.0010, 0.025)` | `-2.07` | A -3% drop is fairly normal in a volatile Bear market |

### Step 2: Numerators (line 45-46)
| Variable | Calculation | Result |
|----------|-------------|--------|
| `unnorm_bull` | $-5.11 + (-0.69)$ | $-5.80$ |
| `unnorm_bear` | $-2.07 + (-0.69)$ | $-2.76$ |

### Step 3: Denominator via `log_sum_exp()` (line 50)
| Sub-step | Calculation | Result |
|----------|-------------|--------|
| Find $c$ | $\max(-5.80, -2.76)$ | $-2.76$ |
| Shifted exponents | $e^{-5.80 - (-2.76)} + e^{-2.76 - (-2.76)}$ | $e^{-3.04} + e^{0}$ = $0.048 + 1.0$ |
| Final | $-2.76 + \ln(1.048)$ | $-2.76 + 0.047 = -2.713$ |

`marginal_log_prob` = **-2.713**

### Step 4: Posteriors (line 53-54)
| Variable | Calculation | Log Result | Human-Readable ($e^{x}$) |
|----------|-------------|------------|---------------------------|
| `posterior_bull` | $-5.80 - (-2.713)$ | $-3.087$ | **4.6% Bull** |
| `posterior_bear` | $-2.76 - (-2.713)$ | $-0.047$ | **95.4% Bear** |

**Result:** After a single -3% tick, the engine's belief shifted from 50/50 to **95.4% Bear Market**. These posteriors are recursively fed as tomorrow's priors.

---

---

## 7. The Data Layer: `DataStream` Class

> **File:** `src/data_stream.py`
> **Concept:** Simulates a live, event-driven market data feed with $O(1)$ memory emission.

Instead of loading an entire CSV into memory at once ($O(N)$), `DataStream` emits returns **one tick at a time**, enforcing the streaming constraint required by the recursive engine.

### Method Reference

| Method / Property | Purpose |
|---|---|
| `__init__(data)` | Accepts a `List[float]` of historical returns. Raises `ValueError` if empty. |
| `next_tick()` → `Optional[float]` | Returns the next return value, or `None` if stream is exhausted. |
| `stream()` → `Iterator[float]` | Python generator for use in `for tick in stream.stream():` loops. |
| `reset()` | Rewinds the internal pointer to tick 0. Useful for re-running backtests. |
| `has_next` (property) | `True` if there are remaining ticks. |
| `progress` (property) | Human-readable string like `"42/500 ticks"`. |

### Usage Pattern
```python
from src.data_stream import DataStream

stream = DataStream([0.01, -0.03, 0.005, -0.02])
for tick in stream.stream():
    # Feed tick into RegimeDetector.update()
    pass
```

---

## 8. The Brain: `RegimeDetector` Class

> **File:** `src/regime_detector.py`
> **Concept:** The main orchestrator class. Wraps all log-space math into a clean interface. Accepts raw market returns and returns human-readable probabilities.

### How It Works Internally

When you call `detector.update(market_return)`, the following happens inside:

```
detector.update(-0.03)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  RegimeDetector.update()                         │
│  File: src/regime_detector.py                    │
│                                                  │
│  Step 1: Likelihoods                             │
│  ├─ log_normal_pdf(x, bull_mean, bull_std)       │
│  └─ log_normal_pdf(x, bear_mean, bear_std)       │
│       ↓ (calls src/math_utils.py, line 20)       │
│                                                  │
│  Step 2: Prepare Priors                          │
│  ├─ IF transition_matrix is set (HMM ON):        │
│  │   Apply HMM Prediction Step via log_sum_exp() │
│  │   (see Section 9 below)                       │
│  └─ ELSE: Use raw priors directly (pure Bayes)   │
│                                                  │
│  Step 3: Posterior                                │
│  └─ compute_posterior_log_space(prior, likelih.)  │
│       ↓ (calls src/math_utils.py, line 30)       │
│                                                  │
│  Step 4: Update State                            │
│  ├─ Store new posteriors as next tick's priors    │
│  ├─ Convert log → decimal via math.exp()         │
│  └─ Append to self._history                      │
│                                                  │
│  Return (bull_prob, bear_prob)  ← floats 0 to 1  │
└──────────────────────────────────────────────────┘
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `bull_mean` | `float` | Mean return ($\mu$) of the Bull regime (e.g., `0.0005`) |
| `bull_std` | `float` | Std dev ($\sigma$) of the Bull regime (e.g., `0.01`) |
| `bear_mean` | `float` | Mean return ($\mu$) of the Bear regime (e.g., `-0.001`) |
| `bear_std` | `float` | Std dev ($\sigma$) of the Bear regime (e.g., `0.025`) |
| `initial_bull_prob` | `float` | Starting belief for Bull (default `0.5` = neutral) |
| `transition_matrix` | `Optional[List[List[float]]]` | 2×2 HMM matrix. `None` = pure Bayes, no dampening. |

### Method Reference

| Method / Property | Purpose |
|---|---|
| `update(market_return)` → `(float, float)` | Processes one tick. Returns `(bull_prob, bear_prob)` as decimals 0–1. |
| `reset()` | Resets beliefs and history for a fresh backtest run. |
| `current_belief` (property) | Returns current `(bull_prob, bear_prob)` without processing a tick. |
| `history` (property) | Full list of `(bull_prob, bear_prob)` for every processed tick. |
| `tick_count` (property) | Number of ticks processed so far. |

### Usage Pattern (Pure Bayes — No Transition Matrix)
```python
from src.data_stream import DataStream
from src.regime_detector import RegimeDetector

detector = RegimeDetector(
    bull_mean=0.0005, bull_std=0.01,
    bear_mean=-0.001, bear_std=0.025,
)
stream = DataStream([0.01, -0.03, 0.005, -0.02])

for tick in stream.stream():
    bull, bear = detector.update(tick)
    print(f"Bull: {bull:.1%}, Bear: {bear:.1%}")
```

### Usage Pattern (With HMM Transition Matrix)
```python
detector = RegimeDetector(
    bull_mean=0.0005, bull_std=0.01,
    bear_mean=-0.001, bear_std=0.025,
    transition_matrix=[
        [0.95, 0.05],  # If Bull today: 95% stay Bull, 5% switch to Bear
        [0.10, 0.90],  # If Bear today: 10% switch to Bull, 90% stay Bear
    ],
)
```

---

## 9. The HMM Transition Matrix (Regime Stickiness)

> **File:** `src/regime_detector.py`, inside `update()` method
> **Concept:** Prevents signal flickering by modeling the fact that market regimes are "sticky" — if you're in a Bull market today, you're very likely still in one tomorrow.

### The Problem It Solves
Without a transition matrix, a single noisy tick (e.g., a flash crash that recovers instantly) would swing the engine to 90% Bear. In reality, regimes persist for weeks or months.

### The Transition Matrix

$$ P(S_{t} | S_{t-1}) = \begin{bmatrix} P(\text{Bull→Bull}) & P(\text{Bull→Bear}) \\ P(\text{Bear→Bull}) & P(\text{Bear→Bear}) \end{bmatrix} = \begin{bmatrix} 0.95 & 0.05 \\ 0.10 & 0.90 \end{bmatrix} $$

### The HMM Prediction Step (inserted before Bayes update)
Before applying the Likelihood, we first adjust our Prior using the transition probabilities. This is the **Forward Algorithm** from Hidden Markov Models:

$$ P(S_t = \text{Bull}) = P(\text{Bull}|\text{Bull}) \cdot P(\text{Bull}_{t-1}) + P(\text{Bull}|\text{Bear}) \cdot P(\text{Bear}_{t-1}) $$
$$ P(S_t = \text{Bear}) = P(\text{Bear}|\text{Bull}) \cdot P(\text{Bull}_{t-1}) + P(\text{Bear}|\text{Bear}) \cdot P(\text{Bear}_{t-1}) $$

In log-space, these multiplications become additions, and the overall sums are computed via `log_sum_exp()`:

```python
# Inside RegimeDetector.update(), when transition_matrix is set:
predicted_bull = log_sum_exp([
    log(P(Bull|Bull)) + log_prior_bull,   # Bull stayed Bull
    log(P(Bull|Bear)) + log_prior_bear,   # Bear switched to Bull
])
predicted_bear = log_sum_exp([
    log(P(Bear|Bull)) + log_prior_bull,   # Bull switched to Bear
    log(P(Bear|Bear)) + log_prior_bear,   # Bear stayed Bear
])
```

These `predicted_bull` and `predicted_bear` values replace the raw priors before Step 3 (the Bayes posterior computation). The effect is that the engine resists sudden regime switches unless the evidence is overwhelmingly strong.

---

## 10. Historical Data Analysis (Phase 4)

To validate the engine against real-world volatility, ShiftEngine utilizes a data extraction layer (`src/fetch_data.py`) to pull historical SPY (S&P 500 ETF) data during critical market regimes.

### Data Storage
Fetched data is stored in the `/data/` directory (ignored by Git) in CSV format:
*   `data/spy_2008_crash.csv`: The Global Financial Crisis (2007–2010).
*   `data/spy_2020_crash.csv`: The COVID-19 Flash Crash (2019–2021).

### Schema Definition

| Column | Description | Mathematical Role |
|---|---|---|
| **Date** | The trading day (with timezone offset). | Indexing and visualization alignment. |
| **Close** | Adjusted Closing Price. | Raw price level for visual context. |
| **Log_Return** | $\ln(\text{Price}_t / \text{Price}_{t-1})$ | **The primary input.** Used as $x$ in `log_normal_pdf()`. |

### Why Log Returns?
In standard finance, simple returns ($(\frac{P_t}{P_{t-1}}) - 1$) are common. However, ShiftEngine uses **Log Returns** for three critical reasons:
1.  **Mathematical Symmetry:** A 10% gain followed by a 10% loss does not return you to 100 in simple math. In log math, $\ln(1.1) + \ln(0.9) \approx 0$, which is more natural for statistical modeling.
2.  **Aggregation:** Log returns are additive over time. The log return over 5 days is simply the sum of daily log returns. 
3.  **PDF Consistency:** The Normal Distribution modeling in `src/math_utils.py` assumes the *change* in price follows a bell curve. Log returns are more likely to be normally distributed than raw price changes.

## 11. The Backtest Engine (Phase 4, Part 2)

> **File:** `src/backtest.py`
> **Concept:** Validates the Bayesian Engine by simulating trading days tick-by-tick using historical CSV data.

The backtest engine loads historical Log Returns, feeds them into the `RegimeDetector` via a `DataStream`, and logs the resulting probability shifts.

### Backtest Configuration
For SPY analysis, the engine is initialized with the following parameters:
*   **Bull State:** $\mu = 0.05\%$, $\sigma = 1.0\%$ (Calm, upward drift)
*   **Bear State:** $\mu = -0.10\%$, $\sigma = 2.5\%$ (Volatile, downward drift)
*   **Transition Matrix:** 98% Bull-persistence, 95% Bear-persistence (High stickiness to prevent flickering).

### Detection Accuracy (Historical Benchmarks)
During the **2020 COVID Crash**, the engine successfully detected a regime shift:
*   **Target Date:** 2020-02-24
*   **Observation:** The engine flipped from BULL to BEAR on the exact morning the market gapped down to start the crash, demonstrating high sensitivity to volatility spikes.

## 12. Visualization (Phase 4, Part 3)

> **File:** `src/visualize_results.py`
> **Concept:** Transitions raw Bayesian data into an intuitive visual format for analysis.

The visualization layer plots the asset price action overlaid with the engine's regime classification and internal probabilities.

### Plot Features
*   **Primary Axis (Left):** SPY Closing Price (Black line). Shows the raw market trend.
*   **Secondary Axis (Right):** Bear Probability (Red dashed line). Shows the engine's internal confidence in a Bear market (0% to 100%).
*   **Color-Coded Backgrounds:**
    *   **Green Shading:** Engine detects a **BULL** regime (Bull Prob > 50%).
    *   **Red Shading:** Engine detects a **BEAR** regime (Bear Prob > 50%).

### Interpretation
A "successful" visualization shows the red shading appearing *before* or *at the exact start* of a major price drop. The 2020 crash plot clearly demonstrates the engine's reaction to volatility on Feb 24th, where the background immediately flips red as the price begins its descent.

---

## 13. Future Scope

### 11.1 Learning Parameters via the Baum-Welch Algorithm
In the current implementation, the transition matrix values (e.g., 0.95 Bull→Bull) and the regime distribution parameters ($\mu$, $\sigma$) are **manually configured** based on domain knowledge.

A natural extension is to use the **Baum-Welch Algorithm** (also known as the **Expectation-Maximization (EM) algorithm for HMMs**) to automatically learn these parameters from historical market data. Given a batch of past returns, Baum-Welch iteratively:

1. **E-Step (Expectation):** Using the current parameter guesses, estimate the probability of being in each regime at every historical time step.
2. **M-Step (Maximization):** Using those estimated regime assignments, re-calculate the optimal $\mu$, $\sigma$, and transition probabilities that best explain the observed data.
3. **Repeat** until the parameters converge (stop changing significantly).

The result: instead of manually guessing "95% Bull→Bull feels right," Baum-Welch would crunch 10 years of S&P 500 data and output the statistically optimal transition matrix and distribution parameters.

**Potential implementation:** Python's `hmmlearn` library provides a ready-made Baum-Welch implementation that could be integrated into the engine's initialization phase to calibrate parameters before the live recursive updates begin.

---

## References
*   [Wikipedia: LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)
*   [Gregory Gundersen's ML Breakdown of the Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
*   [Wikipedia: Baum-Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)

