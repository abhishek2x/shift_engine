# ShiftEngine

**ShiftEngine** is a high-performance filtering tool designed to calculate the probability of being in a "High Volatility/Bear" vs. "Low Volatility/Bull" market state in real-time. It leverages recursive Bayesian updates (Log-Likelihood) and acts as an Online Hidden Markov Model (HMM).

For in-depth mathematical derivations and architecture specifics, see the [Technical Details](docs/technical_details.md).

## 🚀 Strategic Applications
1. **Directly Applicable Integration:** Serves as the foundational signal layer for Trend Following and Dynamic Asset Allocation strategies.
2. **Computational Robustness:** Engineered focusing on numerical safety and low-latency recursive updating in log-space.
3. **Regime Adaptability:** Enables portfolios to dynamically scale volatility exposure (e.g., reducing beta when the probability of a High Volatility state exceeds 80%).

## 🗂️ Project Structure
* `/docs`: Detailed documentation and mathematical specifications.
* `/notebooks`: Prototyping math components and backtesting visualization.
* `/src`: Core OOP logic (`RegimeDetector`, `DataStream`, math utilities).
* `/tests`: Benchmarking and testing scripts ensuring logic precision and $O(1)$ memory compliance.
* `/data`: Local storage for historical simulation data.
