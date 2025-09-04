import sys
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Scenario selection based on command-line argument
arg = sys.argv[1].lower() if len(sys.argv) > 1 else "baseline"
if "tariff" in arg:
    SCENARIO = "TariffScenarioExtended"
else:
    SCENARIO = "Baseline"

print(f"\nRunning forecast for scenario: {SCENARIO}\n")

# File paths and targets
Y_PATH      = "prepared_cash_flow_data.csv"
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

# Load historical target variable data
df_y = pd.read_csv(Y_PATH)
for c in ("ds","statement_date","balance_date","Unnamed: 0"):
    if c in df_y.columns:
        df_y["ds"] = pd.to_datetime(df_y[c])
        break
df_y = df_y[["ds"] + targets]

# Load baseline economic indicators
df_base = (
    pd.read_csv(BASE_IND, parse_dates=["Date"])
      .rename(columns={"Date":"ds"})
)
df_base["ds"] = df_base["ds"] + pd.offsets.MonthEnd(0)
df_base = df_base[["ds"] + econ_regs].sort_values("ds").reset_index(drop=True)

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
if SCENARIO == "TariffScenarioExtended":
    # Add a binary flag for the tariff period as a regressor
    df_full["tariff_flag"] = ((df_full.ds >= T0) & (df_full.ds <= T1)).astype(int)
    regs.append("tariff_flag")

df_master = df_full.merge(df_y, on="ds", how="left")

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
    if "tariff_flag" in regs:
        m.add_regressor("tariff_flag", mode="multiplicative", prior_scale=300.0)

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
fn = f"{SCENARIO}_forecasts_ALL_COMPONENTS.csv"
all_out.to_csv(fn, index=False)
print(f"\nSaved all forecasts to: {fn}\n")