import os
import sys

# Add the project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_stream import DataStream
from src.regime_detector import RegimeDetector

def run_backtest(csv_path: str, label: str):
    """
    Executes a market regime backtest on historical data.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Data file {csv_path} not found. Run fetch_data.py first.")
        return

    print(f"\n{'='*50}")
    print(f"Starting Backtest: {label}")
    print(f"{'='*50}")

    # 1. Load Data
    df = pd.read_csv(csv_path)
    dates = df['Date'].tolist()
    returns = df['Log_Return'].tolist()
    prices = df['Close'].tolist()
    
    # 2. Initialize Engine
    # Standard SPY parameters:
    # Bull: ~12% annual return, ~16% annual vol
    # Bear: ~-25% annual return, ~40% annual vol
    detector = RegimeDetector(
        bull_mean=0.0005,  
        bull_std=0.010,
        bear_mean=-0.0010,
        bear_std=0.025,
        transition_matrix=[
            [0.98, 0.02], # Extremely sticky Bull
            [0.05, 0.95]  # Sticky Bear
        ]
    )
    
    stream = DataStream(returns)
    
    # 3. Simulate Tick-by-Tick
    results = []
    
    print(f"Processing {len(returns)} trading days...")
    
    for i, tick in enumerate(stream.stream()):
        bull_prob, bear_prob = detector.update(tick)
        
        # Determine current state based on 50% threshold
        state = "BULL" if bull_prob > 0.5 else "BEAR"
        
        results.append({
            "Date": dates[i],
            "Price": prices[i],
            "Return": tick,
            "Bull_Prob": bull_prob,
            "Bear_Prob": bear_prob,
            "Regime": state
        })

    # 4. Results Summary
    results_df = pd.DataFrame(results)
    bear_days = len(results_df[results_df['Regime'] == 'BEAR'])
    total_days = len(results_df)
    
    print(f"\nBacktest Complete.")
    print(f"Total Days: {total_days}")
    print(f"Detected Bear Days: {bear_days} ({bear_days/total_days:.1%})")
    
    # Show internal transitions (Signal detection)
    transitions = results_df[results_df['Regime'] != results_df['Regime'].shift(1)]
    print(f"\nDetected {len(transitions)-1} Regime Shifts:")
    for _, row in transitions.iloc[1:11].iterrows(): # Show first 10 shifts
        print(f"  - {row['Date'].split(' ')[0]}: Switched to {row['Regime']}")
    
    if len(transitions) > 11:
        print("    ...")

    # Build absolute path to save results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "..", "data", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    out_file = f"results_{label.lower().replace(' ', '_')}.csv"
    results_df.to_csv(os.path.join(results_dir, out_file), index=False)
    print(f"\n✅ Detailed results saved to /data/results/{out_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    # Run backtests on our two datasets
    run_backtest(
        os.path.join(data_dir, "spy_2008_crash.csv"), 
        "2008 Financial Crisis"
    )
    
    run_backtest(
        os.path.join(data_dir, "spy_2020_crash.csv"), 
        "2020 COVID Crash"
    )
