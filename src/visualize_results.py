import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(base_dir))

def visualize_regimes(results_path: str, title: str, output_name: str):
    if not os.path.exists(results_path):
        print(f"Error: Results file {results_path} not found.")
        return

    # 1. Load Results
    df = pd.read_csv(results_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Setup Plot
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Primary Axis: SPY Price
    ax1.set_xlabel('Date')
    ax1.set_ylabel('SPY Adjusted Close ($)', color='black')
    ax1.plot(df['Date'], df['Price'], color='black', linewidth=1.5, label='SPY Price')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # 3. Add Regime Backgrounds
    # Green for Bull, Red for Bear
    for i in range(len(df) - 1):
        color = 'green' if df['Regime'].iloc[i] == 'BULL' else 'red'
        ax1.axvspan(df['Date'].iloc[i], df['Date'].iloc[i+1], color=color, alpha=0.15)

    # 4. Secondary Axis: Bear Probability
    ax2 = ax1.twinx()
    ax2.set_ylabel('Bear Probability (%)', color='red')
    ax2.plot(df['Date'], df['Bear_Prob'], color='red', alpha=0.4, linestyle='--', label='Bear Probability')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='red')

    # Formatting
    plt.title(f"Market Regime Detection: {title}", fontsize=14, pad=20)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    # Create manual legends to combine both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Ensure images directory exists
    img_dir = os.path.join(os.path.dirname(base_dir), "images")
    os.makedirs(img_dir, exist_ok=True)
    
    plt.tight_layout()
    save_path = os.path.join(img_dir, output_name)
    plt.savefig(save_path)
    print(f"✅ Visualization saved to /images/{output_name}")
    plt.close()

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(base_dir), "data", "results")
    
    # Visualize both crashes
    visualize_regimes(
        os.path.join(results_dir, "results_2008_financial_crisis.csv"),
        "2008 Financial Crisis",
        "regime_detection_2008.png"
    )
    
    visualize_regimes(
        os.path.join(results_dir, "results_2020_covid_crash.csv"),
        "2020 COVID Crash",
        "regime_detection_2020.png"
    )
