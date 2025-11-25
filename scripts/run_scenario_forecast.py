import sys
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

# Scenario selection based on command-line argument
arg = sys.argv[1].lower() if len(sys.argv) > 1 else "baseline"
if "tariff" in arg:
    SCENARIO = "TariffScenarioExtended"
else:
    SCENARIO = "Baseline"

print(f"\nRunning forecast for scenario: {SCENARIO}\n")

# File paths and targets
Y_PATH      = "prepared_cash_flow_data.csv"
SYNTHETIC_HISTORY_PATH = Path("data/synthetic_historical_training_data.csv")
USE_SYNTHETIC_HISTORY = True  # blends synthetic history with recent Valentis data if available
BASE_IND    = "mock_economic_indicators.csv"
TARIFF_IND  = "mock_economic_indicators_TARIFF_SCENARIO_EXTENDED.csv"

targets     = ["inflow_operating_revenue", "outflow_cogs", "outflow_opex"]
econ_regs   = [
    "Economic_Activity_Index",
    "Consumer_Confidence_Index",
    "Interest_Rate_Proxy",
    "EV_Component_Cost_Index"
]

# Define the window for the tariff scenario
T0 = pd.to_datetime("2025-02-28") + pd.offsets.MonthEnd(0)
T1 = pd.to_datetime("2026-06-30") + pd.offsets.MonthEnd(0)

# Load historical target variable data (blended with synthetic history)
# We want to use Synthetic History (2000-2021) to learn long-term economic relationships
# and Real Valentis Data (2021-Present) to learn recent scale and seasonality.

# 1. Load Real Valentis Data
df_y_actual = pd.read_csv(Y_PATH)
for c in ("ds","statement_date","balance_date","Unnamed: 0"):
    if c in df_y_actual.columns:
        df_y_actual["ds"] = pd.to_datetime(df_y_actual[c])
        break

# 2. Load Synthetic History
df_syn = pd.read_csv(SYNTHETIC_HISTORY_PATH, parse_dates=["ds"])
# Ensure columns match for concatenation
df_syn = df_syn.rename(columns={"y": "inflow_operating_revenue"})

# 3. Combine Them
# Filter synthetic to end strictly before Valentis data starts to avoid overlap
valentis_start_date = df_y_actual['ds'].min()
df_syn = df_syn[df_syn['ds'] < valentis_start_date].copy()

# Add placeholder columns for other targets in synthetic data (set to NaN or 0)
# Since synthetic data only models 'inflow_operating_revenue', other targets
# will effectively rely only on recent history or need their own synthetic proxies.
# For now, we leave them as NaN so Prophet ignores them during those periods
# (Prophet handles missing y values).
# Ensure synthetic dataframe has the columns we need; if not, fill NaN (only for missing ones)
# Since we updated build_synthetic_history.py, these should now exist for COGS and Opex.
for tgt in targets:
    if tgt not in df_syn.columns:
        print(f"Warning: {tgt} not found in synthetic history. Filling with NaN.")
        df_syn[tgt] = np.nan

# Select only necessary columns from both
df_syn = df_syn[["ds"] + targets].copy() # Indicators will come from df_full later
df_y_actual_reduced = df_y_actual[["ds"] + targets].copy()

# Concatenate
df_y = pd.concat([df_syn, df_y_actual_reduced], axis=0, ignore_index=True).sort_values("ds").reset_index(drop=True)

print(f"Combined History Loaded: {len(df_y)} months (Synthetic: {len(df_syn)}, Actual: {len(df_y_actual)})")

# Load baseline economic indicators
# We need indicators for the FULL history (2000-Present) to match df_y
# 1. Load Synthetic Indicators (2000-2021)
df_syn_ind = pd.read_csv(SYNTHETIC_HISTORY_PATH, parse_dates=["ds"])
df_syn_ind = df_syn_ind[["ds"] + econ_regs].copy()

# 2. Load Real Baseline Indicators (2021-Present)
df_base_real = (
    pd.read_csv(BASE_IND, parse_dates=["Date"])
      .rename(columns={"Date":"ds"})
)
df_base_real["ds"] = df_base_real["ds"] + pd.offsets.MonthEnd(0)
df_base_real = df_base_real[["ds"] + econ_regs].sort_values("ds").reset_index(drop=True)

# 3. Combine Indicators
# Filter synthetic to end before real data starts
real_start = df_base_real["ds"].min()
df_syn_ind = df_syn_ind[df_syn_ind["ds"] < real_start].copy()

df_base = pd.concat([df_syn_ind, df_base_real], axis=0, ignore_index=True).sort_values("ds").reset_index(drop=True)
print(f"Combined Indicators Loaded: {len(df_base)} months (Synthetic: {len(df_syn_ind)}, Real: {len(df_base_real)})")

# If tariff scenario, overlay the mock-tariff values during the defined window
if SCENARIO == "TariffScenarioExtended":
    df_tar = (
        pd.read_csv(TARIFF_IND, parse_dates=["Date"])
          .rename(columns={"Date":"ds"})
    )
    df_tar["ds"] = df_tar["ds"] + pd.offsets.MonthEnd(0)
    df_tar = df_tar[["ds"] + econ_regs].sort_values("ds").reset_index(drop=True)

    # Merge base and tariff data, then select tariff values only within the window
    df_full = pd.merge(df_base, df_tar,
                       on="ds", how="left",
                       suffixes=("_base","_tariff"))
    mask = (df_full.ds >= T0) & (df_full.ds <= T1)
    for r in econ_regs:
        df_full[r] = np.where(mask,
                              df_full[f"{r}_tariff"],
                              df_full[f"{r}_base"])
    df_full = df_full[["ds"] + econ_regs]

    # Sanity check to compare indicator values
    mb = df_base.loc[(df_base.ds>=T0)&(df_base.ds<=T1), ["ds"]+econ_regs]
    mt = df_tar. loc[(df_tar. ds>=T0)&(df_tar. ds<=T1), ["ds"]+econ_regs]
    df_cmp = mb.rename(columns={r:f"{r}_base" for r in econ_regs}) \
               .merge(mt.rename(columns={r:f"{r}_tariff" for r in econ_regs}),
                      on="ds")
    print("Tariff vs. Baseline indicator samples:")
    print(df_cmp.head(6), "\n")

else:
    df_full = df_base.copy()

# Build regressor list and master table for modeling
regs = econ_regs.copy()
# Note: We rely purely on economic indicators (Interest Rate, Confidence, etc.)
# to drive the forecast. No hardcoded 'tariff_flag' is used.

# Use outer join to keep all history (df_y) even if df_full was missing some dates (though it shouldn't now)
df_master = df_full.merge(df_y, on="ds", how="outer").sort_values("ds")
# Drop rows where we have no regressors (if any)
df_master = df_master.dropna(subset=econ_regs)

# Forecast loop for each target variable
all_out = None
for tgt in targets:
    print(f"\nForecasting {tgt}")
    df_t      = df_master[["ds"] + regs + [tgt]].rename(columns={tgt:"y"})
    df_hist   = df_t[df_t.y.notna()].copy()
    train, test = df_hist.iloc[:-6], df_hist.iloc[-6:]

    # Initialize and configure Prophet model
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )

    # Add economic indicators as multiplicative regressors
    for r in econ_regs:
        m.add_regressor(r, mode="multiplicative", prior_scale=100.0)

    # Add tariff flag as a multiplicative regressor if applicable
    # (Removed to rely on economic indicators)

    m.fit(train)

    # Print out the learned regressor coefficients
    betas = m.params["beta"].mean(axis=0)
    names = ["intercept"] + list(m.extra_regressors.keys())
    print("Regressor coefficients:")
    for n,b in zip(names,betas):
        print(f"  {n:30s} {b:8.4f}")
    print()

    # Evaluate model performance on the hold-out test set
    ptest = m.predict(test[["ds"] + regs])
    print(" MAE on last 6 months:", 
          mean_absolute_error(test.y, ptest.yhat))

    # Generate the future forecast
    future = m.make_future_dataframe(periods=24, freq="ME")
    future = future.merge(df_full, on="ds", how="left")
    fc     = m.predict(future)

    # Plot the forecast and save the figure
    fig = m.plot(fc)
    ax  = fig.gca()
    ax.plot(test.ds, test.y, "r.", label="hold-out data")
    fig.suptitle(f"{tgt} ({SCENARIO})", y=0.92)
    fig.savefig(f"plot_{tgt}_{SCENARIO}.png")
    plt.close(fig)

    # Collect and combine the forecast results
    out = pd.DataFrame({
        "ds":             fc.ds,
        f"{tgt}_forecast": fc.yhat,
        f"{tgt}_lower":    fc.yhat_lower,
        f"{tgt}_upper":    fc.yhat_upper,
    })
    actuals = df_t[["ds","y"]].rename(columns={"y":f"{tgt}_actual"})
    out     = actuals.merge(out, on="ds", how="outer").sort_values("ds")
    all_out = out if all_out is None else all_out.merge(out, on="ds", how="outer")

    # Save all combined forecasts to a single CSV file
    out_name = "baseline" if SCENARIO.lower() == "baseline" else SCENARIO
    fn = f"{out_name}_forecasts_ALL_COMPONENTS.csv"
all_out.to_csv(fn, index=False)
print(f"\nSaved all forecasts to: {fn}\n")
