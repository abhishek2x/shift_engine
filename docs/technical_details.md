# ShiftEngine Technical Details

This document outlines the core mathematical executions of the ShiftEngine algorithm, specifically outlining how formulas translate into continuous, safe Python calculations.

---

## 1. The Recursive Bayesian Update Flow
*(Referenced in `src/math_utils.py` -> `compute_posterior_log_space()`)*

When a new S&P 500 return arrives from the `DataStream` (e.g., $x_{t} = -0.03$ or a -3% drop), the recursive engine executes the following steps entirely in Log-Space.

### Step 1: Calculate the Log-Likelihoods
*(Referenced in `src/math_utils.py` -> `log_normal_pdf()`)*

We invoke the same Probability Density Function (PDF) but ask two different questions based on configured market regimes:
1.  **Call 1 (Bull Market):** "What are the odds of seeing a -3% drop if the average return is purely positive ($+0.05\%$) and volatility is low ($1.0\%$)?"
    *   `log_likelihood_bull = log_normal_pdf(-0.03, 0.0005, 0.01)` $\approx -3.50$
2.  **Call 2 (Bear Market):** "What are the odds of seeing a -3% drop if the average return is negative ($-0.10\%$) and volatility is wild ($2.5\%$)?"
    *   `log_likelihood_bear = log_normal_pdf(-0.03, -0.0010, 0.025)` $\approx -1.20$

### Step 2: Apply Priors (The Numerator)
In Bayes' Theorem, the Numerator is: $\text{Likelihood} \times \text{Prior}$.
Since logarithms transform multiplication into addition ($\ln(A \times B) = \ln(A) + \ln(B)$), we safely bypass CPU decimal precision limits:
*   `unnorm_bull` = `log_likelihood_bull` $+$ `prior_bull`
*   `unnorm_bear` = `log_likelihood_bear` $+$ `prior_bear`

### Step 3: Calculate the Marginal Likelihood (The Denominator)
*(Referenced in `src/math_utils.py` -> `log_sum_exp()`)*

Bayes' Theorem requires dividing by the **Marginal Likelihood** (the absolute probability of seeing that exact -3% return regardless of the market state). This requires adding the probabilities of the separate states together:
$P(\text{Data}) = P(\text{Data}|\text{Bull}) + P(\text{Data}|\text{Bear})$

Because our numbers are locked in logs, we cannot add them ($ \ln(A) + \ln(B) \neq \ln(A+B) $). To do this without crashing the program via Floating Point Underflow, we utilize the **Log-Sum-Exp (LSE) Trick** (Explained in detail below). We pass the `[unnorm_bull, unnorm_bear]` variables array into this function.

*   `marginal_log_prob = log_sum_exp([unnorm_bull, unnorm_bear])`

### Step 4: Normalize (The Posterior)
We finalize Bayes' Theorem by dividing our Numerator by the Denominator. In log-space, division becomes **subtraction**:
*   `posterior_bull` = `unnorm_bull` $-$ `marginal_log_prob`
*   `posterior_bear` = `unnorm_bear` $-$ `marginal_log_prob`

These updated probabilities instantly become the Priors for the next incoming data tick, closing the $O(1)$ memory loop!

---

## 2. In-Depth: The Log-Sum-Exp Trick
If we try to add two extremely small floating-point decimals, the CPU rounds them to exactly `0.0`. Taking $\ln(0.0)$ crashes the engine due to a Math Domain Error. The **Log-Sum-Exp (LSE) Trick** is an algebraic workaround to add the underlying log numbers without exponentiating them back into extreme decimals.

**The Base Problem:**
Calculate $\ln(e^A + e^B)$ without underflowing. Let $A = -100$ and $B = -105$. Python cannot directly calculate $e^{-100}$.

**The Algebraic Solution:**
1.  Find the absolute maximum value: Let $c = \max(A, B)$. Here, $c = -100$.
2.  Factor $e^c$ cleanly out of the equation using exponent rules ($ e^A + e^B = e^c \cdot ( e^{A-c} + e^{B-c} ) $).
3.  Inject this back into the logarithm:
    $$ \ln \left[ e^c \cdot ( e^{A-c} + e^{B-c} ) \right] $$
4.  By log multiplication rules ($\ln(X \cdot Y) = \ln(X) + \ln(Y)$), split the logarithm:
    $$ \ln(e^c) + \ln( e^{A-c} + e^{B-c} ) $$
5.  Since $\ln$ and $e$ instantly cancel each other out on the left variable, we arrive at the perfect computation trick:
    $$ c + \ln( e^{A-c} + e^{B-c} ) $$

Because we subtracted $c$ from the exponents inside the parenthesis *before* taking the $e$, the integers inside the natural log are small, positive, and completely safe for python to handle (e.g., $\ln(e^0 + e^{-5})$).

**Additional Reading:**
*   [Wikipedia: LogSumExp](https://en.wikipedia.org/wiki/LogSumExp)
*   [Machine Learning Mathematical Proof: The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
