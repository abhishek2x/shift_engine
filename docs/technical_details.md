# ShiftEngine \- Technical Mathematics & Architecture Guide

This document breaks down the mathematical foundation of ShiftEngine, explicitly mapping the theoretical concepts to the actual variables and functions inside the Python codebase.

---

## 1. The Core Equation: Bayes' Theorem
At its heart, ShiftEngine executes recursive Bayes' Theorem.
$$ P(State_{t} | Return_{t}) = \frac{P(Return_{t} | State_{t}) \cdot P(State_{t-1})}{P(Return_{t})} $$

### Definitions
1.  **Prior Probability** $P(State_{t-1})$: Your belief *before* seeing the new market data.
2.  **Likelihood** $P(Return_{t} | State_{t})$: The probability of standard normal distributions generating the specific market data we just saw.
3.  **Marginal Likelihood (Evidence)** $P(Return_{t})$: **This is the denominator!** This is the total, absolute probability of seeing that exact market return, regardless of what state the market is in. It requires us to *add* the probability of observing the data in a Bull market to the probability of observing it in a Bear market.
4.  **Posterior** $P(State_{t} | Return_t)$: Your computed, updated belief. This will be recursively passed down as tomorrow's Prior.

---

## 2. The Computational Danger: Floating Point Underflow
In recursive Bayesian updates, you continuously multiply extremely small probability decimals (e.g., $0.05 \times 0.01 \times 0.20 \dots$). 

Very quickly, these chained multiplications result in a number smaller than a CPU's floating-point precision limit (typically $10^{-320}$). When this ceiling is breached, the program undergoes **Underflow** and rounds the probability to exactly `0.0`. Once a probability hits zero, it can never multiply back up. 

---

## 3. The Implementation Flow (`src/math_utils.py`)
To prevent Underflow, we operate entirely in **Log-Space**, transforming multiplication into simple addition: $\ln(A \times B) = \ln(A) + \ln(B)$.

When a new S&P 500 return arrives from your `DataStream` class, the core logic invokes `compute_posterior_log_space()`:

### Step 1: Calculate the Likelihoods
We call `log_normal_pdf(x, mean, std_dev)` twice. We pass $x$ (the market return) against two completely different configurations.
*   **Bull Check:** We ask, "What are the odds of seeing this return if the market is trending up with low volatility?" (Outputs `log_likelihood_bull`)
*   **Bear Check:** We ask, "What are the odds of seeing this return if the market is trending down with massive volatility?" (Outputs `log_likelihood_bear`)

### Step 2: Combine Prior & Likelihood (The Numerators)
Because we are in log-space, we simply **add** our Prior log-beliefs to our new Likelihoods.
*   `unnorm_bull` = `log_likelihood_bull` + `prior_log_bull`
*   `unnorm_bear` = `log_likelihood_bear` + `prior_log_bear`

### Step 3: Calculate Marginal Likelihood (The Denominator & The Log-Sum-Exp Trick)
We must complete Bayes' Theorem by dividing by the Evidence/Marginal Likelihood. The Marginal Likelihood demands we **add** our newly calculated Bull and Bear probabilities together.

But we are in log-space, and **you cannot add logs together** ($\ln(A) + \ln(B) \neq \ln(A + B)$). Mathematically, we have to un-log them by exponentiating them back to decimals ($e^A + e^B$) before adding them.

**The Crash:** Executing $e^A$ instantly triggers an Underflow Crash.
**The Solution:** We pass the array `[unnorm_bull, unnorm_bear]` into the `log_sum_exp()` function.

**Algebraic Derivation of LSE Trick:**
Let $c$ be the largest log-number (e.g., $c = \max(A, B)$).
1.  Start with: $\ln(e^A + e^B)$
2.  Factor out $e^c$: $\ln \left[ e^c \cdot (e^{A-c} + e^{B-c}) \right]$
3.  Split the log multiplication: $\ln(e^c) + \ln(e^{A-c} + e^{B-c})$
4.  Cancel the $\ln(e)$ on the left: $c + \ln(e^{A-c} + e^{B-c})$

*Because we subtracted $c$ from the exponents before executing $e$, the exponents are small ($0$ and a very tiny negative number). Underflow is perfectly avoided, and computations are flawless.*

*   The output of `log_sum_exp()` becomes our denominator, `marginal_log_prob`.

### Step 4: Normalize to Posteriors
Finally, we calculate the continuous Bayes Posterior. Since we are in log-space, division becomes **subtraction**.
*   `posterior_bull` = `unnorm_bull` - `marginal_log_prob`
*   `posterior_bear` = `unnorm_bear` - `marginal_log_prob`

These two values are recursively stored to be used as `prior_log_probs` during the next clock tick!

---
*For further reading on the mathematics of the LSE trick, refer to:*
*   [Wikipedia: LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)
*   [Gregory Gundersen's ML Breakdown of the Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
