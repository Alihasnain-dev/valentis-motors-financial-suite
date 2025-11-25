import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pathlib import Path
from typing import Dict

# Configuration
START_DATE = "2000-01-01"
END_DATE = "2024-01-01"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "synthetic_historical_training_data.csv"

# FRED series to fetch via pandas_datareader
FRED_SERIES = {
    "Interest_Rate_Proxy": "FEDFUNDS",
    "Consumer_Confidence_Index": "UMCSENT",
    "Economic_Activity_Index": "INDPRO",
    "Total_Vehicle_Sales": "TOTALSA",
    "CPI": "CPIAUCSL",
}

# Target scaling: auto-size to Valentis current average revenue if available
PREPARED_DATA_PATH = Path("prepared_cash_flow_data.csv")
TARGET_SCALE_FALLBACK = 500_000_000  # used only if prepared file missing

NOISE_STD = 0.02  # 5% noise to avoid perfect correlation


def fetch_fred_data(series_map: Dict[str, str]) -> pd.DataFrame:
    # Invert map for DataReader: it expects list of FRED IDs
    fred_ids = list(series_map.values())
    inv_map = {v: k for k, v in series_map.items()}
    
    print(f"Fetching from FRED: {fred_ids}")
    df = web.DataReader(fred_ids, "fred", START_DATE, END_DATE)
    
    # Rename columns to our internal names
    df.rename(columns=inv_map, inplace=True)
    
    # Resample to monthly start and forward fill
    df = df.resample("MS").mean().ffill()
    
    # Reset index to make 'ds' a column
    df = df.reset_index().rename(columns={"DATE": "ds"})
    return df


def get_real_data_stats():
    """Calculates the starting scale and average ratios from the prepared data."""
    if PREPARED_DATA_PATH.exists():
        df = pd.read_csv(PREPARED_DATA_PATH)
        # Sort by date to get the start
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'ds' in c), None)
        if date_col:
            df = df.sort_values(date_col)
            # Take the first 12 months of REAL data to calibrate the end of Synthetic data
            start_slice = df.head(12)
            
            avg_rev = start_slice['inflow_operating_revenue'].mean()
            
            # Calculate ratios based on real history
            if 'outflow_cogs' in df.columns:
                avg_cogs_ratio = (start_slice['outflow_cogs'] / start_slice['inflow_operating_revenue']).mean()
            else:
                avg_cogs_ratio = 0.65
                
            if 'outflow_opex' in df.columns:
                avg_opex_ratio = (start_slice['outflow_opex'] / start_slice['inflow_operating_revenue']).mean()
            else:
                avg_opex_ratio = 0.15
                
            return avg_rev, avg_cogs_ratio, avg_opex_ratio
            
    return 500_000_000, 0.65, 0.15 # Fallbacks

def build_synthetic(df_macro: pd.DataFrame, scale_rev: float, ratio_cogs: float, ratio_opex: float) -> pd.DataFrame:
    current_market_scale = df_macro["Total_Vehicle_Sales"].iloc[-12:].mean()
    scaling_factor = scale_rev / current_market_scale
    df = df_macro.copy()
    
    np.random.seed(42)
    noise_rev = np.random.normal(0, NOISE_STD, len(df))
    
    # 1. Revenue
    df["inflow_operating_revenue"] = df["Total_Vehicle_Sales"] * scaling_factor * (1 + noise_rev)
    
    # 2. Cost Indices
    df["EV_Component_Cost_Index"] = df["CPI"]
    # NORMALIZE the index so it centers around 1.0 (or the end value)
    # This ensures the multiplier works mathematically as a percentage scaler
    cost_index_normalized = df["EV_Component_Cost_Index"] / df["EV_Component_Cost_Index"].mean()
    
    # 3. COGS Calculation (The Critical Fix)
    # COGS = Revenue * Base_Ratio * Cost_Index_Factor
    # If Cost Index goes up 10%, COGS Ratio goes up 10%
    df["outflow_cogs"] = df["inflow_operating_revenue"] * ratio_cogs * cost_index_normalized
    
    # 4. Opex (Apply Real Ratio)
    df["outflow_opex"] = df["inflow_operating_revenue"] * ratio_opex

    return df[["ds", "inflow_operating_revenue", "outflow_cogs", "outflow_opex", "Interest_Rate_Proxy", "Consumer_Confidence_Index", "EV_Component_Cost_Index", "Economic_Activity_Index"]]

def main():
    print(f"Fetching FRED series: {FRED_SERIES}")
    df_macro = fetch_fred_data(FRED_SERIES)
    
    rev_scale, cogs_ratio, opex_ratio = get_real_data_stats()
    print(f"Calibrating Synthetic Data: Rev Scale=${rev_scale:,.0f}, COGS Ratio={cogs_ratio:.2f}, Opex Ratio={opex_ratio:.2f}")
    
    df_syn = build_synthetic(df_macro, rev_scale, cogs_ratio, opex_ratio)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote synthetic history to {OUTPUT_FILE}")
if __name__ == "__main__":
    main()
