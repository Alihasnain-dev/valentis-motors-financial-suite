import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import numpy as np
import os

# Database Connection Details
DB_NAME = "valentis_motors_db"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("VALENTIS_DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"

# File Paths
capex_file_path = "Monthly_Mock_CapEx.csv"
indicators_file_path = "mock_economic_indicators.csv"
prepared_data_output_path = "prepared_cash_flow_data.csv"

print("Starting Cash Flow Data Preparation Process...")

# 1. Define SQL Queries for Cash Flow Components
sql_queries = {
    "income_statement": """
        SELECT
            statement_date,
            revenue,
            cogs_materials,
            cogs_labor,
            cogs_overhead,
            operating_expenses,
            interest_expense,
            taxes
        FROM income_statement_data
        ORDER BY statement_date;
    """,
    "balance_sheet_for_debt": """
        SELECT
            balance_date,
            long_term_debt,
            short_term_debt
        FROM balance_sheet_data
        ORDER BY balance_date;
    """
}

# 2. Connect to PostgreSQL and Fetch Data
conn_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
historical_dfs = {}

try:
    engine = create_engine(conn_string)
    for key, query in sql_queries.items():
        print(f"Fetching data for: {key}")
        historical_dfs[key] = pd.read_sql(query, engine)
        date_col = 'statement_date' if 'statement_date' in historical_dfs[key].columns else 'balance_date'
        historical_dfs[key][date_col] = pd.to_datetime(historical_dfs[key][date_col])
        historical_dfs[key].set_index(date_col, inplace=True)
        print(f"Finished fetching {key}. Shape: {historical_dfs[key].shape}")

except Exception as e:
    print(f"Error connecting to database or fetching historical data: {e}")

# 3. Load Mock CapEx and Economic Indicators from CSVs
try:
    print(f"\nLoading CapEx data from: {capex_file_path}")
    df_capex = pd.read_csv(capex_file_path)
    df_capex['Date'] = pd.to_datetime(df_capex['Date'])
    df_capex.set_index('Date', inplace=True)
    df_capex.rename(columns={'CapEx_Amount': 'capex_outflow'}, inplace=True)
    df_capex = df_capex.resample('ME').sum()

    print(f"\nLoading Economic Indicators from: {indicators_file_path}")
    df_indicators = pd.read_csv(indicators_file_path)
    df_indicators['Date'] = pd.to_datetime(df_indicators['Date'])
    df_indicators.set_index('Date', inplace=True)
    df_indicators = df_indicators.resample('ME').mean()

except FileNotFoundError as e:
    print(f"ERROR: CSV file not found. {e}")
    exit()
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit()

# 4. Assemble the Main DataFrame
if "income_statement" not in historical_dfs:
    print("CRITICAL ERROR: Income statement data not loaded. Exiting.")
    exit()

df_main = historical_dfs["income_statement"].copy()

# Merge CapEx data
df_main = pd.merge(df_main, df_capex[['capex_outflow']], left_index=True, right_index=True, how='left')
df_main['capex_outflow'] = df_main['capex_outflow'].fillna(0)

# Merge Economic Indicators data
df_main = pd.merge(df_main, df_indicators, left_index=True, right_index=True, how='left')
indicator_cols = df_indicators.columns.tolist()
df_main[indicator_cols] = df_main[indicator_cols].ffill()

# Calculate and Merge Net Debt Activity, and include original debt columns
if "balance_sheet_for_debt" in historical_dfs:
    df_debt = historical_dfs["balance_sheet_for_debt"].copy()
    
    df_debt['long_term_debt'] = pd.to_numeric(df_debt['long_term_debt'], errors='coerce').fillna(0)
    df_debt['short_term_debt'] = pd.to_numeric(df_debt['short_term_debt'], errors='coerce').fillna(0)
    
    df_debt['total_debt'] = df_debt['long_term_debt'] + df_debt['short_term_debt']
    df_debt['net_debt_financing_activity'] = df_debt['total_debt'].diff().fillna(0)
    
    cols_to_merge_from_debt = ['long_term_debt', 'short_term_debt', 'net_debt_financing_activity']
    df_main = pd.merge(df_main, df_debt[cols_to_merge_from_debt], left_index=True, right_index=True, how='left')
    
    for col in cols_to_merge_from_debt:
        if col in df_main.columns:
             df_main[col] = df_main[col].fillna(0)
else:
    print("WARNING: Balance sheet data for debt not found.")
    df_main['long_term_debt'] = 0
    df_main['short_term_debt'] = 0
    df_main['net_debt_financing_activity'] = 0

# 5. Calculate Net Cash Flow Components
# Ensure all component columns exist and are clean before calculations
required_is_cols = ['revenue', 'cogs_materials', 'cogs_labor', 'cogs_overhead', 'operating_expenses', 'interest_expense', 'taxes']
for col in required_is_cols:
    if col not in df_main.columns:
        print(f"Warning: Column '{col}' not found in income statement data. Treating as 0.")
        df_main[col] = 0
    else:
        df_main[col] = df_main[col].fillna(0)

# Calculate Cash Inflow
df_main['inflow_operating_revenue'] = df_main['revenue']

# Calculate Cash Outflows
df_main['outflow_cogs'] = df_main['cogs_materials'] + df_main['cogs_labor'] + df_main['cogs_overhead']
df_main['outflow_opex'] = df_main['operating_expenses']
df_main['outflow_interest'] = df_main['interest_expense']
df_main['outflow_taxes'] = df_main['taxes']

# Calculate Net Operating Cash Flow (CFO)
df_main['net_operating_cash_flow'] = (df_main['inflow_operating_revenue'] -
                                     (df_main['outflow_cogs'] +
                                      df_main['outflow_opex'] +
                                      df_main['outflow_interest'] +
                                      df_main['outflow_taxes']))

# Calculate Net Investing Cash Flow (CFI)
df_main['net_investing_cash_flow'] = -df_main['capex_outflow']

# Calculate Net Financing Cash Flow (CFF)
df_main['net_financing_cash_flow'] = df_main['net_debt_financing_activity']

# Calculate Overall Net Change in Cash
df_main['net_change_in_cash_total'] = (df_main['net_operating_cash_flow'] +
                                     df_main['net_investing_cash_flow'] +
                                     df_main['net_financing_cash_flow'])

print("\nMain Assembled DataFrame Head:")
print(df_main.head())
print("\nMain Assembled DataFrame Info:")
df_main.info()

# 6. Save Prepared Data
try:
    df_main.to_csv(prepared_data_output_path)
    print(f"\nSuccessfully saved prepared data to: {prepared_data_output_path}")
except Exception as e:
    print(f"Error saving prepared data: {e}")

print("\nCash Flow Data Preparation Process Completed.")