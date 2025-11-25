import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASELINE_COMPONENTS_PATH = "baseline_forecasts_ALL_COMPONENTS.csv"
TARIFF_COMPONENTS_PATH = "TariffScenarioExtended_forecasts_ALL_COMPONENTS.csv"
HISTORICAL_PREPARED_DATA_PATH = "prepared_cash_flow_data.csv"
BASE_INDICATORS_PATH = "mock_economic_indicators.csv"
TARIFF_INDICATORS_PATH = "mock_economic_indicators_TARIFF_SCENARIO_EXTENDED.csv"

# Output File Paths
BASELINE_SUMMARY_PATH = "final_cash_flow_summary_BASELINE.csv"
TARIFF_SUMMARY_PATH = "final_cash_flow_summary_TARIFF.csv"
COMPARISON_PLOT_PATH = "plot_scenario_comparison_revenue.png"

TARIFF_START_DATE = pd.to_datetime("2025-02-28")
TARIFF_END_DATE = pd.to_datetime("2026-06-30")

ELASTICITY_ACTIVITY = 0.6    # Revenue sensitivity to activity index pct change
ELASTICITY_CONFIDENCE = 0.4  # Revenue sensitivity to confidence pct change
ELASTICITY_COST = 0.6        # COGS sensitivity to component cost pct change

print("Starting Scenario Build & Comparison (using model outputs)...")


def load_forecast_file(path: str, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=["ds"])
        print(f"{label} component forecasts loaded from {path}.")
        return df
    except Exception as e:
        print(f"FATAL: Could not load {label} forecast file '{path}': {e}")
        raise


def merge_actuals_and_forecasts(df_components: pd.DataFrame) -> pd.DataFrame:
    """Use actuals when present, otherwise forecast."""
    df = df_components.copy()
    for target in ["inflow_operating_revenue", "outflow_cogs", "outflow_opex"]:
        actual_col = f"{target}_actual"
        forecast_col = f"{target}_forecast"
        final_col = f"{target}_final"
        if actual_col not in df.columns or forecast_col not in df.columns:
            raise ValueError(f"Expected columns missing for {target}")
        df[final_col] = df[actual_col].fillna(df[forecast_col])
    return df


def calculate_aggregated_cash_flow(df_input: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    print(f"\nCalculating Aggregated Cash Flow for: {scenario_name}")
    df = df_input.copy()

    # Load historical non-forecasted fields
    df_hist_raw = pd.read_csv(HISTORICAL_PREPARED_DATA_PATH)
    date_col_hist = None
    if "ds" in df_hist_raw.columns:
        date_col_hist = "ds"
    elif "statement_date" in df_hist_raw.columns:
        date_col_hist = "statement_date"
    elif "Unnamed: 0" in df_hist_raw.columns and pd.api.types.is_datetime64_any_dtype(
        pd.to_datetime(df_hist_raw["Unnamed: 0"], errors="coerce")
    ):
        date_col_hist = "Unnamed: 0"
    elif pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_hist_raw.iloc[:, 0], errors="coerce")):
        date_col_hist = df_hist_raw.columns[0]
    if not date_col_hist:
        raise ValueError("Could not find date column in historical data file.")

    df_hist_raw[date_col_hist] = pd.to_datetime(df_hist_raw[date_col_hist])
    if date_col_hist != "ds":
        df_hist_raw.rename(columns={date_col_hist: "ds"}, inplace=True)

    last_actual_ltd = df_hist_raw.get("long_term_debt", pd.Series(dtype=float)).dropna().iloc[-1]
    cols_to_merge = ["ds", "interest_expense", "taxes", "net_debt_financing_activity", "capex_outflow"]
    existing_cols = [c for c in cols_to_merge if c in df_hist_raw.columns]
    df = pd.merge(df, df_hist_raw[existing_cols], on="ds", how="left")

    # Interest/taxes assumptions
    df["long_term_debt_projected"] = last_actual_ltd
    df["long_term_debt_projected"] = df["long_term_debt_projected"].ffill()
    df["interest_forecast"] = (df["long_term_debt_projected"].fillna(0) * 0.03) / 12
    df["interest_final"] = df["interest_expense"].fillna(df["interest_forecast"])

    df["ebit_final"] = df["inflow_operating_revenue_final"] - (
        df["outflow_cogs_final"] + df["outflow_opex_final"]
    )
    df["ebt_final"] = df["ebit_final"] - df["interest_final"]
    df["taxes_forecast"] = np.where(df["ebt_final"] > 0, df["ebt_final"] * 0.25, 0)
    df["taxes_final"] = df["taxes"].fillna(df["taxes_forecast"])

    df["net_operating_cash_flow"] = (
        df["inflow_operating_revenue_final"]
        - (
            df["outflow_cogs_final"]
            + df["outflow_opex_final"]
            + df["interest_final"]
            + df["taxes_final"]
        )
    )
    df["net_investing_cash_flow"] = -df["capex_outflow"].fillna(0)
    df["net_financing_cash_flow"] = df["net_debt_financing_activity"].fillna(0)
    df["net_change_in_cash_total"] = (
        df["net_operating_cash_flow"]
        + df["net_investing_cash_flow"]
        + df["net_financing_cash_flow"]
    )
    print(f"Finished calculations for {scenario_name}.")
    return df


baseline_components = merge_actuals_and_forecasts(load_forecast_file(BASELINE_COMPONENTS_PATH, "Baseline"))
tariff_components = merge_actuals_and_forecasts(load_forecast_file(TARIFF_COMPONENTS_PATH, "Tariff"))

# Deterministic elasticity overlay for tariff revenue/COGS based on indicator deltas
try:
    df_base_ind = pd.read_csv(BASE_INDICATORS_PATH, parse_dates=["Date"]).rename(columns={"Date": "ds"})
    df_tar_ind = pd.read_csv(TARIFF_INDICATORS_PATH, parse_dates=["Date"]).rename(columns={"Date": "ds"})
    df_ind = df_base_ind.merge(df_tar_ind, on="ds", suffixes=("_base", "_tariff"))
    for col in [
        "Economic_Activity_Index",
        "Consumer_Confidence_Index",
        "EV_Component_Cost_Index",
    ]:
        if f"{col}_base" not in df_ind.columns or f"{col}_tariff" not in df_ind.columns:
            raise KeyError(f"Missing indicator columns for {col}")
    df_ind["activity_pct_delta"] = (
        df_ind["Economic_Activity_Index_tariff"] - df_ind["Economic_Activity_Index_base"]
    ) / df_ind["Economic_Activity_Index_base"]
    df_ind["confidence_pct_delta"] = (
        df_ind["Consumer_Confidence_Index_tariff"] - df_ind["Consumer_Confidence_Index_base"]
    ) / df_ind["Consumer_Confidence_Index_base"]
    df_ind["cost_pct_delta"] = (
        df_ind["EV_Component_Cost_Index_tariff"] - df_ind["EV_Component_Cost_Index_base"]
    ) / df_ind["EV_Component_Cost_Index_base"]
    tariff_components = tariff_components.merge(df_ind[["ds", "activity_pct_delta", "confidence_pct_delta", "cost_pct_delta"]], on="ds", how="left")
    # Only adjust forecasted periods (where actuals are NaN)
    forecast_mask_rev = tariff_components["inflow_operating_revenue_actual"].isna()
    rev_adj = 1 + (
        ELASTICITY_ACTIVITY * tariff_components["activity_pct_delta"].fillna(0)
        + ELASTICITY_CONFIDENCE * tariff_components["confidence_pct_delta"].fillna(0)
    )
    tariff_components.loc[forecast_mask_rev, "inflow_operating_revenue_final"] *= rev_adj.loc[forecast_mask_rev]

    # COGS Strategic Override removed.
    # The model now organically predicts higher COGS due to the 'shocked' input cost index.
    # We rely on the Prophet model's learned sensitivity to EV_Component_Cost_Index.
    
    print("Skipping manual COGS override (Model now handles Cost sensitivity organically).")
    print("Applied tariff elasticity overlay to revenue.")
except Exception as e:
    print(f"WARNING: Could not apply elasticity overlay, continuing without it: {e}")

df_baseline_final = calculate_aggregated_cash_flow(baseline_components, "Baseline")
df_tariff_final = calculate_aggregated_cash_flow(tariff_components, "TariffScenarioExtended")

df_baseline_final.to_csv(BASELINE_SUMMARY_PATH, index=False)
df_tariff_final.to_csv(TARIFF_SUMMARY_PATH, index=False)
print(f"\nSaved summaries -> {BASELINE_SUMMARY_PATH} and {TARIFF_SUMMARY_PATH}")

print("\nGenerating comparison plot...")
plt.figure(figsize=(15, 8))
plt.plot(df_baseline_final["ds"], df_baseline_final["inflow_operating_revenue_final"], label="Baseline Revenue", color="blue")
plt.plot(df_tariff_final["ds"], df_tariff_final["inflow_operating_revenue_final"], label="Tariff Revenue", color="red", linestyle="--")
actuals = df_baseline_final[df_baseline_final["inflow_operating_revenue_actual"].notna()]
plt.plot(actuals["ds"], actuals["inflow_operating_revenue_actual"], "k.", label="Historical Actuals")
plt.title("Baseline vs. Tariff Scenario: Operating Revenue")
plt.xlabel("Date")
plt.ylabel("Operating Revenue")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.axvspan(TARIFF_START_DATE, TARIFF_END_DATE, color="red", alpha=0.1, label="Tariff Period")
plt.legend()
plt.savefig(COMPARISON_PLOT_PATH)
plt.close()
print(f"Saved Comparison Plot -> {COMPARISON_PLOT_PATH}")

print("\nProcess Completed.")
