# ShiftEngine

**ShiftEngine** is a high-performance filtering tool designed to calculate the probability of being in a "High Volatility/Bear" vs. "Low Volatility/Bull" market state in real-time. It leverages recursive Bayesian updates (Log-Likelihood) and acts as an Online Hidden Markov Model (HMM).

## 🛠️ Technical Overview

### 1. The Mathematical Core
Instead of batch processing historical datasets ($O(N)$ memory), ShiftEngine processes data continuously with **$O(1)$ memory complexity** via a **Recursive Update**:

**Bayes' Theorem (**$\mathbf{Recursive Form}$**):**

$$ P(State_{t} | Return_{t}) = \frac{P(Return_{t} | State_{t}) \cdot P(State_{t-1})}{P(Return_{t})} $$

Where:
*   **Prior ($P(State_{t-1})$):** Your current foundational belief of the market state.
*   **Likelihood ($P(Return_{t} | State_t)$):** How well a newly observed return fits configured "Bull" vs "Bear" normal distributions.
*   **Posterior ($P(State_{t} | Return_t)$):** Your computed, updated belief. 

**Normal Distribution Likelihood:**

$$ P(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) $$

**The Log-Space Transformation:**
To prevent floating-point underflow across long time series (a critical numerical stability requirement), all calculations are performed in log-space. The natural log formulation prevents the decimals from decaying to zero:

$$ \ln P(x|\mu, \sigma) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2} $$

**The Log-Sum-Exp Trick:**
Because the denominator in Bayes' theorem requires the marginal probability (adding the probabilities of being in either state), and our values are now in log-space, we cannot simply add logs together. We calculate the sum securely by utilizing the Log-Sum-Exp trick:

$$ \ln(e^A + e^B) = c + \ln(e^{A-c} + e^{B-c}) $$
*(where $c = \max(A, B)$)*

### 2. The System Architecture
*   **Event-Driven Ingestion:** Class-based structures (`DataStream`) simulate live market data point-by-point, bridging the gap between backtesting and a production execution environment.
*   **HMM Transition Matrix:** Implements state hysteresis to dampen noise.
$$ P(S_{t} | S_{t-1}) \approx \begin{bmatrix} 0.95 & 0.05 \\ 0.10 & 0.90 \end{bmatrix} $$

## 🚀 Strategic Applications
1.  **Directly Applicable Integration:** Serves as the foundational signal layer for Trend Following and Dynamic Asset Allocation strategies.
2.  **Computational Robustness:** Engineered focusing on numerical safety and low-latency recursive updating.
3.  **Regime Adaptability:** Enables portfolios to dynamically scale volatility exposure (e.g., reducing beta when the probability of a High Volatility state exceeds 80%).

## 🗂️ Project Structure (Roadmap)
*   `/notebooks`: Prototyping math components and backtesting visualization.
*   `/src`: Core OOP logic (`RegimeDetector`, `DataStream`, math utilities).
*   `/tests`: Benchmarking and testing scripts ensuring logic precision and $O(1)$ memory compliance.
*   `/data`: Local storage for historical simulation data.
