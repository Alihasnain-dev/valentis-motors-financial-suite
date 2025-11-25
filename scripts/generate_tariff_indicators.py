import pandas as pd
import numpy as np

# Configuration
BASE_INDICATORS_PATH = "mock_economic_indicators.csv"
OUTPUT_PATH = "mock_economic_indicators_TARIFF_SCENARIO_EXTENDED.csv"

# Tariff Settings
TARIFF_START_DATE = "2025-03-01"
TARIFF_END_DATE = "2026-06-30"
TARIFF_IMPACT_ON_COSTS = 0.25  # 25% increase in component costs due to tariffs
CONFIDENCE_SHOCK = -0.15       # 15% drop in consumer confidence

print("Generating Shocked Economic Indicators for Tariff Scenario...")

try:
    # Load Baseline
    df = pd.read_csv(BASE_INDICATORS_PATH, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)

    # Define the Mask for the Tariff Period
    mask = (df['Date'] >= TARIFF_START_DATE) & (df['Date'] <= TARIFF_END_DATE)

    # 1. Shock the Component Cost Index (The Driver for COGS)
    # We apply a multiplier to the existing trend
    # This effectively tells the model: "Inputs are 25% more expensive now."
    if 'EV_Component_Cost_Index' in df.columns:
        df.loc[mask, 'EV_Component_Cost_Index'] = df.loc[mask, 'EV_Component_Cost_Index'] * (1 + TARIFF_IMPACT_ON_COSTS)
    else:
        print("Warning: 'EV_Component_Cost_Index' column not found. Skipping cost shock.")

    # 2. Shock Consumer Confidence (The Driver for Revenue)
    # Confidence drops, signaling lower demand
    if 'Consumer_Confidence_Index' in df.columns:
        df.loc[mask, 'Consumer_Confidence_Index'] = df.loc[mask, 'Consumer_Confidence_Index'] * (1 + CONFIDENCE_SHOCK)
    else:
        print("Warning: 'Consumer_Confidence_Index' column not found. Skipping confidence shock.")

    # 3. (Optional) Shock Economic Activity if you want deeper recession logic
    # df.loc[mask, 'Economic_Activity_Index'] = df.loc[mask, 'Economic_Activity_Index'] * 0.95

    # Save the new scenario file
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated tariff scenario indicators with:")
    print(f" - {TARIFF_IMPACT_ON_COSTS*100}% increase in EV Component Costs")
    print(f" - {abs(CONFIDENCE_SHOCK)*100}% decrease in Consumer Confidence")
    print(f"Saved to: {OUTPUT_PATH}")

except FileNotFoundError:
    print(f"Error: Base indicators file not found at {BASE_INDICATORS_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")