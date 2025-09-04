import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BASELINE_COMPONENTS_PATH = "baseline_forecasts_ALL_COMPONENTS.csv"
HISTORICAL_PREPARED_DATA_PATH = "prepared_cash_flow_data.csv"

# Output File Paths
BASELINE_SUMMARY_PATH = "final_cash_flow_summary_BASELINE.csv"
TARIFF_SUMMARY_PATH = "final_cash_flow_summary_TARIFF.csv"
COMPARISON_PLOT_PATH = "plot_scenario_comparison_revenue.png"

# Tariff Scenario Parameters
TARIFF_START_DATE = pd.to_datetime('2025-02-28')
TARIFF_END_DATE = pd.to_datetime('2026-06-30')
TARIFF_REVENUE_IMPACT = -0.15  # 15% reduction in revenue
TARIFF_COGS_IMPACT = 0.10      # 10% increase in COGS

print("Starting Scenario Building Process...")

# Load Baseline Forecast Data
try:
    df_baseline = pd.read_csv(BASELINE_COMPONENTS_PATH, parse_dates=['ds'])
    print("Baseline component forecasts loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load baseline forecast file '{BASELINE_COMPONENTS_PATH}'. {e}")
    exit()

# Create the primary dataframe for calculations
df_final = df_baseline.copy()

# Create final component columns that use actuals when available, otherwise use forecast
for target in ["inflow_operating_revenue", "outflow_cogs", "outflow_opex"]:
    df_final[f'{target}_final'] = df_final[f'{target}_actual'].fillna(df_final[f'{target}_forecast'])

# Build the Tariff Scenario
print("\nBuilding Tariff Scenario based on Baseline Forecast...")
df_tariff = df_final.copy()

# Define the mask for the tariff period
tariff_period_mask = (df_tariff['ds'] >= TARIFF_START_DATE) & (df_tariff['ds'] <= TARIFF_END_DATE)

# Apply the direct impacts to the '_final' columns for the tariff period
df_tariff.loc[tariff_period_mask, 'inflow_operating_revenue_final'] *= (1 + TARIFF_REVENUE_IMPACT)
df_tariff.loc[tariff_period_mask, 'outflow_cogs_final'] *= (1 + TARIFF_COGS_IMPACT)

print(f"Applied {TARIFF_REVENUE_IMPACT*100:.0f}% revenue impact and +{TARIFF_COGS_IMPACT*100:.0f}% COGS impact during tariff period.")

# Function to Calculate Full Cash Flow from Components
def calculate_aggregated_cash_flow(df_input, scenario_name):
    print(f"\nCalculating Aggregated Cash Flow for: {scenario_name}")
    df_scenario_calc = df_input.copy()

    try:
        # Load historical data to source non-forecasted actuals
        df_hist_full_raw = pd.read_csv(HISTORICAL_PREPARED_DATA_PATH)
        date_col_hist = None
        if 'ds' in df_hist_full_raw.columns: date_col_hist = 'ds'
        elif 'statement_date' in df_hist_full_raw.columns: date_col_hist = 'statement_date'
        elif 'Unnamed: 0' in df_hist_full_raw.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_hist_full_raw['Unnamed: 0'], errors='coerce')):
            date_col_hist = 'Unnamed: 0'
        elif pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_hist_full_raw.iloc[:, 0], errors='coerce')):
            date_col_hist = df_hist_full_raw.columns[0]
        if not date_col_hist: raise ValueError("Could not find date column in historical data file.")
        
        df_hist_full = df_hist_full_raw.copy()
        df_hist_full[date_col_hist] = pd.to_datetime(df_hist_full[date_col_hist])
        if date_col_hist != 'ds': df_hist_full.rename(columns={date_col_hist: 'ds'}, inplace=True)
        
        # Get last known debt for interest projection
        last_actual_ltd = df_hist_full['long_term_debt'].dropna().iloc[-1]
    except Exception as e:
        print(f"FATAL ERROR loading historical data for calculations in {scenario_name}: {e}")
        return None

    # Merge historical actuals into the scenario dataframe
    cols_to_merge = ['ds', 'interest_expense', 'taxes', 'net_debt_financing_activity', 'capex_outflow']
    existing_cols_to_merge = [col for col in cols_to_merge if col in df_hist_full.columns]
    df_scenario_calc = pd.merge(df_scenario_calc, df_hist_full[existing_cols_to_merge], on='ds', how='left')
    
    # Project Interest and Taxes for the forecast period
    df_scenario_calc['long_term_debt_projected'] = last_actual_ltd
    df_scenario_calc['long_term_debt_projected'].ffill(inplace=True)
    
    df_scenario_calc['interest_forecast'] = (df_scenario_calc['long_term_debt_projected'].fillna(0) * 0.03) / 12
    df_scenario_calc['interest_final'] = df_scenario_calc['interest_expense'].fillna(df_scenario_calc['interest_forecast'])

    df_scenario_calc['ebit_final'] = df_scenario_calc['inflow_operating_revenue_final'] - (df_scenario_calc['outflow_cogs_final'] + df_scenario_calc['outflow_opex_final'])
    df_scenario_calc['ebt_final'] = df_scenario_calc['ebit_final'] - df_scenario_calc['interest_final']
    df_scenario_calc['taxes_forecast'] = np.where(df_scenario_calc['ebt_final'] > 0, df_scenario_calc['ebt_final'] * 0.25, 0)
    df_scenario_calc['taxes_final'] = df_scenario_calc['taxes'].fillna(df_scenario_calc['taxes_forecast'])

    # Calculate Final Aggregated Cash Flows
    df_scenario_calc['net_operating_cash_flow'] = (
        df_scenario_calc['inflow_operating_revenue_final'] -
        (df_scenario_calc['outflow_cogs_final'] + df_scenario_calc['outflow_opex_final'] + df_scenario_calc['interest_final'] + df_scenario_calc['taxes_final'])
    )
    df_scenario_calc['net_investing_cash_flow'] = -df_scenario_calc['capex_outflow'].fillna(0)
    df_scenario_calc['net_financing_cash_flow'] = df_scenario_calc['net_debt_financing_activity'].fillna(0)

    df_scenario_calc['net_change_in_cash_total'] = (
        df_scenario_calc['net_operating_cash_flow'] +
        df_scenario_calc['net_investing_cash_flow'] +
        df_scenario_calc['net_financing_cash_flow']
    )
    
    print(f"Finished calculations for {scenario_name}.")
    return df_scenario_calc

# Process and Save Both Scenarios
df_baseline_final = calculate_aggregated_cash_flow(df_final, "Baseline")
df_tariff_final = calculate_aggregated_cash_flow(df_tariff, "TariffScenario")

if df_baseline_final is not None and df_tariff_final is not None:
    df_baseline_final.to_csv(BASELINE_SUMMARY_PATH, index=False)
    print(f"\nSaved Baseline Summary -> {BASELINE_SUMMARY_PATH}")
    df_tariff_final.to_csv(TARIFF_SUMMARY_PATH, index=False)
    print(f"Saved Tariff Scenario Summary -> {TARIFF_SUMMARY_PATH}")

    # Plot Comparison Chart
    print("\nGenerating comparison plot...")
    plt.figure(figsize=(15, 8))
    
    # Plot final revenue for both scenarios
    plt.plot(df_baseline_final['ds'], df_baseline_final['inflow_operating_revenue_final'], label='Baseline Revenue Forecast', color='blue', linestyle='-')
    plt.plot(df_tariff_final['ds'], df_tariff_final['inflow_operating_revenue_final'], label='Tariff Scenario Revenue Forecast', color='red', linestyle='--')
    
    # Plot historical actuals for context
    actuals = df_baseline_final[df_baseline_final['inflow_operating_revenue_actual'].notna()]
    plt.plot(actuals['ds'], actuals['inflow_operating_revenue_actual'], 'k.', label='Historical Actuals')
    
    plt.title('Baseline vs. Tariff Scenario: Forecasted Operating Revenue')
    plt.xlabel('Date')
    plt.ylabel('Operating Revenue')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvspan(TARIFF_START_DATE, TARIFF_END_DATE, color='red', alpha=0.1, label='Tariff Period')
    plt.legend()
    plt.savefig(COMPARISON_PLOT_PATH)
    plt.close()
    print(f"Saved Comparison Plot -> {COMPARISON_PLOT_PATH}")

print("\nProcess Completed.")