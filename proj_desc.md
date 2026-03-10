# Project Description

**The Goal:** Build a high-performance filtering tool that calculates the probability of being in a "High Volatility/Bear" vs. "Low Volatility/Bull" market state in real-time using recursive Bayesian updates (Log-Likelihood).

For specific technical implementation, transition matrices, and mathematical formulas, please refer to [Technical Details](docs/technical_details.md).

---

## 📋 The Feature List

- **Recursive Updates:** Process market data sequentially with $O(1)$ memory.
- **Log-Space Processing:** To avoid floating-point underflow (a critical numerical stability requirement), perform all calculations in **Log-Space**.
- **Dual-Distribution Modeling:** Define two regimes:
  ![image.png](./images/image1.png)
- **The "Hysteresis" Filter:** Add a "Transition Matrix" (Ross Ch 4/Markov concepts). This represents the probability that if we are in a Bull market today, we stay in it tomorrow (e.g., 95%). This prevents the signal from "flickering" on every small price move.
- **Data Ingestion:** Create a class-based structure that accepts a "stream" of returns (simulated or historical).
- **Visualizer:** A plot showing the price series on top and the "Probability of Bear Market" (0 to 1) on the bottom.

---

## 🚀 Strategic Applications

1. **Directly Applicable Integration:** This engine serves as the foundational layer for **Trend Following** and **Dynamic Asset Allocation** strategies.
2. **Computational Robustness:** It demonstrates the ability to safely implement continuous-time continuous-space mathematics (log-probabilities, $O(1)$ memory management) in a production environment.
3. **Regime Shift Adaptability:** By dynamically tracking regime shifts using an Online Hidden Markov Model (HMM), the engine rapidly adapts portfolio exposure without relying on lagging indicators like moving averages.
