import pandas as pd
import pandas_datareader.data as web
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
SYNTHETIC_HISTORY_PATH = Path("data/synthetic_historical_training_data.csv")
PREPARED_DATA_PATH = Path("prepared_cash_flow_data.csv")
BACKTEST_START_DATE = "2023-01-01"
CURRENT_DATE = "2024-11-01" # Approximation of "now" for data fetching

FRED_SERIES = {
    "Interest_Rate_Proxy": "FEDFUNDS",
    "Consumer_Confidence_Index": "UMCSENT",
    "Economic_Activity_Index": "INDPRO",
    "Total_Vehicle_Sales": "TOTALSA",
    "CPI": "CPIAUCSL",
}

def fetch_real_world_benchmark():
    """Fetches 2023-2024 real data to serve as the 'Truth'."""
    print(f"Fetching Real-World Benchmark Data (2020-{CURRENT_DATE})...")
    fred_ids = list(FRED_SERIES.values())
    inv_map = {v: k for k, v in FRED_SERIES.items()}
    
    # Fetch ample history to ensure we can calculate scale correctly
    df = web.DataReader(fred_ids, "fred", "2020-01-01", CURRENT_DATE)
    df.rename(columns=inv_map, inplace=True)
    df = df.resample("MS").mean().ffill().reset_index().rename(columns={"DATE": "ds"})
    
    # Infer Scaling Factor from the stored prepared data (Valentis Scale)
    # matching the logic in build_synthetic_history.py
    if PREPARED_DATA_PATH.exists():
        df_prep = pd.read_csv(PREPARED_DATA_PATH)
        # Try to find a revenue column
        rev_col = next((c for c in df_prep.columns if "revenue" in c), None)
        if rev_col:
            valentis_scale = df_prep[rev_col].mean()
        else:
            valentis_scale = 500_000_000
    else:
        valentis_scale = 500_000_000
        
    # Get Market Scale (using 2023 data as reference or just the last 12 months of the fetch)
    current_market_scale = df["Total_Vehicle_Sales"].iloc[-12:].mean()
    scaling_factor = valentis_scale / current_market_scale
    
    print(f"Valentis Scale: ${valentis_scale:,.0f}")
    print(f"Scaling Factor: {scaling_factor}")
    
    # Create the Benchmark Revenue Line
    df["Benchmark_Revenue"] = df["Total_Vehicle_Sales"] * scaling_factor
    
    # Use CPI for Cost Index
    df["EV_Component_Cost_Index"] = df["CPI"]
    
    return df

def run_backtest():
    print("\n--- Starting Holdout Validation Test (Backtest) ---")
    
    # 1. Load Training Data (Synthetic History)
    if not SYNTHETIC_HISTORY_PATH.exists():
        print("Error: Synthetic history file not found. Run build_synthetic_history.py first.")
        return

    df_train = pd.read_csv(SYNTHETIC_HISTORY_PATH, parse_dates=["ds"])
    
    # Map specific target to 'y' for Prophet
    if "inflow_operating_revenue" in df_train.columns:
        df_train["y"] = df_train["inflow_operating_revenue"]
    
    # CUTOFF: We strictly hide any data after Jan 1, 2023 from the model
    # The model must predict 2023 and 2024 knowing ONLY 2000-2022.
    train_mask = df_train["ds"] < BACKTEST_START_DATE
    df_train_cutoff = df_train.loc[train_mask].copy()
    
    print(f"Training Data Cutoff: {df_train_cutoff['ds'].max()}")
    print(f"Training Rows: {len(df_train_cutoff)}")
    
    # 2. Setup Prophet Model
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    
    econ_regs = [
        "Economic_Activity_Index",
        "Consumer_Confidence_Index",
        "Interest_Rate_Proxy",
        "EV_Component_Cost_Index"
    ]
    
    for r in econ_regs:
        m.add_regressor(r, mode="multiplicative")
        
    m.fit(df_train_cutoff)
    
    # 3. Prepare Future Data (The "Test Set")
    # We need the ACTUAL economic indicators for 2023-2024 to feed the model
    # We get these from the fresh FRED fetch
    df_truth = fetch_real_world_benchmark()
    
    # Filter for the test period
    test_mask = df_truth["ds"] >= BACKTEST_START_DATE
    df_test_future = df_truth.loc[test_mask].copy()
    
    if len(df_test_future) == 0:
        print("Error: No data fetched for 2023-2024. Check internet connection or FRED.")
        return

    print(f"Forecasting {len(df_test_future)} months (2023-2024)...")
    
    # Predict
    forecast = m.predict(df_test_future)
    
    # 4. Compare & Visualize
    df_comparison = df_test_future[["ds", "Benchmark_Revenue"]].merge(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds"
    )
    
    # Calculate Error
    mape = mean_absolute_percentage_error(df_comparison["Benchmark_Revenue"], df_comparison["yhat"])
    print(f"\nBACKTEST RESULTS:")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2%}")
    
    if mape < 0.10:
        print("Verdict: EXCELLENT. Model tracks reality closely.")
    elif mape < 0.20:
        print("Verdict: ACCEPTABLE. Model captures trends but has variance.")
    else:
        print("Verdict: VOLATILE. Model is over-sensitive or miss-calibrated.")
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_train_cutoff["ds"].iloc[-24:], df_train_cutoff["y"].iloc[-24:], 'k.', label="Training Data (2021-2022)")
    plt.plot(df_comparison["ds"], df_comparison["Benchmark_Revenue"], 'g-', linewidth=2, label="Real Benchmark (Truth)")
    plt.plot(df_comparison["ds"], df_comparison["yhat"], 'b--', linewidth=2, label="Model Forecast")
    plt.fill_between(df_comparison["ds"], df_comparison["yhat_lower"], df_comparison["yhat_upper"], color='b', alpha=0.2)
    
    plt.title(f"Model Validation Backtest (2023-2024)\nMAPE: {mape:.1%}")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "validation_backtest_2023_2024.png"
    plt.savefig(out_file)
    print(f"Saved validation plot to: {out_file}")

if __name__ == "__main__":
    run_backtest()