# ShiftEngine

**ShiftEngine** is a high-performance filtering tool designed to calculate the probability of being in a "High Volatility/Bear" vs. "Low Volatility/Bull" market state in real-time. It leverages recursive Bayesian updates (Log-Likelihood) and acts as an Online Hidden Markov Model (HMM).

## 🛠️ Technical Overview

### 1. The Mathematical Core
Instead of batch processing historical datasets ($O(N)$ memory), ShiftEngine processes data continuously with **$O(1)$ memory complexity** via a **Recursive Update**:
*   **Prior:** Your current foundational belief of the market state.
*   **Likelihood:** How well a newly observed return ($r_{t+1}$) fits configured "Bull" vs "Bear" normal distributions.
*   **Posterior:** Your numerically stable, updated belief. 

To prevent floating-point underflow across long time series (a critical numerical stability requirement), all Bayesian additions are handled using the **Log-Sum-Exp Trick** in log-space.

### 2. The System Architecture
*   **Event-Driven Ingestion:** Class-based structures (`DataStream`) simulate live market data point-by-point, bridging the gap between backtesting and a production execution environment.
*   **HMM Transition Matrix:** Implements state hysteresis (e.g., $P(\text{Bull} | \text{Bull}) = 0.95$) to dampen noise and prevent rapid "signal flickering" on standard market noise.

## 🚀 Strategic Applications
1.  **Directly Applicable Integration:** Serves as the foundational signal layer for Trend Following and Dynamic Asset Allocation strategies.
2.  **Computational Robustness:** Engineered focusing on numerical safety and low-latency recursive updating.
3.  **Regime Adaptability:** Enables portfolios to dynamically scale volatility exposure (e.g., automatically reducing beta allocation when the probability of a High Volatility state exceeds 80%).

## 🗂️ Project Structure (Roadmap)
*   `/notebooks`: Prototyping math components and backtesting visualization.
*   `/src`: Core OOP logic (`RegimeDetector`, `DataStream`, math utilities).
*   `/tests`: Benchmarking and testing scripts ensuring logic precision and $O(1)$ memory compliance.
*   `/data`: Local storage for historical simulation data.
