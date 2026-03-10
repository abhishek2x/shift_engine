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

## References
*   [Wikipedia: LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)
*   [Gregory Gundersen's ML Breakdown of the Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
