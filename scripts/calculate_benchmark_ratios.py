import pandas as pd
import psycopg2
import numpy as np
import os

# Database Connection Details
DB_NAME = "valentis_motors_db"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("VALENTIS_DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"

# Helper to safely perform division for ratio calculations
def safe_division(numerator, denominator):
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    if pd.isna(numerator):
        return np.nan 
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError):
        return np.nan

# Ratio Calculation Functions
def calculate_profitability_ratios(row_dict):
    ratios = []
    gpm = safe_division(row_dict.get('gross_profit'), row_dict.get('total_revenue'))
    ratios.append({'ratio_category': 'Profitability', 'ratio_name': 'Gross Profit Margin', 'ratio_value': gpm})
    opm = safe_division(row_dict.get('operating_income'), row_dict.get('total_revenue'))
    ratios.append({'ratio_category': 'Profitability', 'ratio_name': 'Operating Profit Margin', 'ratio_value': opm})
    npm = safe_division(row_dict.get('net_income'), row_dict.get('total_revenue'))
    ratios.append({'ratio_category': 'Profitability', 'ratio_name': 'Net Profit Margin', 'ratio_value': npm})
    roa = safe_division(row_dict.get('net_income'), row_dict.get('total_assets'))
    ratios.append({'ratio_category': 'Profitability', 'ratio_name': 'Return on Assets (ROA)', 'ratio_value': roa})
    roe = safe_division(row_dict.get('net_income'), row_dict.get('total_equity'))
    ratios.append({'ratio_category': 'Profitability', 'ratio_name': 'Return on Equity (ROE)', 'ratio_value': roe})
    return ratios

def calculate_liquidity_ratios(row_dict):
    ratios = []
    cr = safe_division(row_dict.get('total_current_assets'), row_dict.get('total_current_liabilities'))
    ratios.append({'ratio_category': 'Liquidity', 'ratio_name': 'Current Ratio', 'ratio_value': cr})
    quick_assets = None
    if pd.notna(row_dict.get('total_current_assets')) and pd.notna(row_dict.get('inventory')):
        quick_assets = row_dict.get('total_current_assets') - row_dict.get('inventory')
    qr = safe_division(quick_assets, row_dict.get('total_current_liabilities'))
    ratios.append({'ratio_category': 'Liquidity', 'ratio_name': 'Quick Ratio (Acid Test)', 'ratio_value': qr})
    return ratios

def calculate_solvency_ratios(row_dict):
    ratios = []
    de = safe_division(row_dict.get('total_liabilities'), row_dict.get('total_equity'))
    ratios.append({'ratio_category': 'Solvency', 'ratio_name': 'Debt-to-Equity Ratio', 'ratio_value': de})
    da = safe_division(row_dict.get('total_liabilities'), row_dict.get('total_assets'))
    ratios.append({'ratio_category': 'Solvency', 'ratio_name': 'Debt-to-Assets Ratio', 'ratio_value': da})
    icr = safe_division(row_dict.get('operating_income'), row_dict.get('interest_expense'))
    ratios.append({'ratio_category': 'Solvency', 'ratio_name': 'Interest Coverage Ratio', 'ratio_value': icr})
    return ratios

def calculate_efficiency_ratios(row_dict):
    ratios = []
    at = safe_division(row_dict.get('total_revenue'), row_dict.get('total_assets'))
    ratios.append({'ratio_category': 'Efficiency', 'ratio_name': 'Asset Turnover', 'ratio_value': at})
    it = safe_division(row_dict.get('cost_of_revenue'), row_dict.get('inventory'))
    ratios.append({'ratio_category': 'Efficiency', 'ratio_name': 'Inventory Turnover', 'ratio_value': it})
    return ratios

def calculate_growth_ratios(current_row_dict, prev_row_dict):
    ratios = []
    prev_revenue = prev_row_dict.get('total_revenue')
    rev_growth = safe_division(current_row_dict.get('total_revenue') - prev_revenue, abs(prev_revenue) if pd.notna(prev_revenue) and prev_revenue != 0 else np.nan)
    ratios.append({'ratio_category': 'Growth', 'ratio_name': 'Revenue Growth YoY', 'ratio_value': rev_growth})
    
    prev_net_income = prev_row_dict.get('net_income')
    ni_growth = safe_division(current_row_dict.get('net_income') - prev_net_income, abs(prev_net_income) if pd.notna(prev_net_income) and prev_net_income != 0 else np.nan)
    ratios.append({'ratio_category': 'Growth', 'ratio_name': 'Net Income Growth YoY', 'ratio_value': ni_growth})
    return ratios

print("Starting Benchmark Ratio Calculation Process...")
all_calculated_ratios_list = []
conn = None
try:
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    # Fetch and Prepare Valentis Motors Annual Data
    print("\nProcessing Valentis Motors data...")
    valentis_is_query = """
    SELECT EXTRACT(YEAR FROM statement_date)::INTEGER AS fiscal_year,
        SUM(revenue) AS total_revenue, SUM(cogs_materials + cogs_labor + cogs_overhead) AS cost_of_revenue,
        SUM(gross_profit) AS gross_profit, SUM(ebit) AS operating_income,
        SUM(interest_expense) AS interest_expense, SUM(net_income) AS net_income
    FROM income_statement_data GROUP BY fiscal_year ORDER BY fiscal_year;"""
    df_valentis_is = pd.read_sql(valentis_is_query, conn)

    valentis_bs_query = """
    SELECT DISTINCT ON (EXTRACT(YEAR FROM balance_date))
        EXTRACT(YEAR FROM balance_date)::INTEGER AS fiscal_year, total_assets,
        (inventory_raw_materials + inventory_wip + inventory_finished_goods) AS inventory,
        accounts_payable, long_term_debt, total_current_assets, total_current_liabilities,
        total_liabilities, equity, cash AS cash_and_equivalents
    FROM balance_sheet_data ORDER BY EXTRACT(YEAR FROM balance_date), balance_date DESC;"""
    df_valentis_bs = pd.read_sql(valentis_bs_query, conn)

    if not df_valentis_is.empty and not df_valentis_bs.empty:
        df_valentis_annual = pd.merge(df_valentis_is, df_valentis_bs, on="fiscal_year", how="inner")
        df_valentis_annual['company_name'] = 'Valentis Motors'
        print(f"Valentis Motors annual data prepared. Shape: {df_valentis_annual.shape}")
        df_valentis_annual_sorted = df_valentis_annual.sort_values(by='fiscal_year')
        for i, row_tuple in enumerate(df_valentis_annual_sorted.itertuples(index=False)):
            row_data_dict = row_tuple._asdict()
            year = row_data_dict['fiscal_year']
            all_calculated_ratios_list.extend([{'company_name':'Valentis Motors','fiscal_year':year,**r} for r in calculate_profitability_ratios(row_data_dict)])
            all_calculated_ratios_list.extend([{'company_name':'Valentis Motors','fiscal_year':year,**r} for r in calculate_liquidity_ratios(row_data_dict)])
            all_calculated_ratios_list.extend([{'company_name':'Valentis Motors','fiscal_year':year,**r} for r in calculate_solvency_ratios(row_data_dict)])
            all_calculated_ratios_list.extend([{'company_name':'Valentis Motors','fiscal_year':year,**r} for r in calculate_efficiency_ratios(row_data_dict)])
            if i > 0:
                prev_row_data_dict = df_valentis_annual_sorted.iloc[i-1].to_dict()
                all_calculated_ratios_list.extend([{'company_name':'Valentis Motors','fiscal_year':year,**r} for r in calculate_growth_ratios(row_data_dict, prev_row_data_dict)])
    else:
        print("WARNING: Valentis Motors historical data incomplete for ratio calculation.")

    # Fetch and Process Competitor Data
    print("\nProcessing Competitor data...")
    competitors_query = "SELECT competitor_id, competitor_name FROM competitors;"
    df_competitors_list = pd.read_sql(competitors_query, conn)

    for _, comp_row in df_competitors_list.iterrows():
        competitor_id, competitor_name = comp_row['competitor_id'], comp_row['competitor_name']
        print(f"Calculating ratios for: {competitor_name}")
        comp_financials_query_string = "SELECT * FROM competitor_annual_financials WHERE competitor_id = %s ORDER BY fiscal_year;"
        df_comp_annual = pd.read_sql(comp_financials_query_string, conn, params=(competitor_id,))
        if df_comp_annual.empty:
            print(f"No data for {competitor_name}. Skipping.")
            continue

        expected_cols_for_ratios = [
            'total_revenue', 'cost_of_revenue', 'gross_profit', 'operating_income', 'net_income', 'interest_expense',
            'total_assets', 'inventory', 'accounts_payable', 'long_term_debt',
            'total_equity', 'total_liabilities', 'total_current_assets', 'total_current_liabilities', 'cash_and_equivalents'
        ]
        for col in expected_cols_for_ratios:
            if col not in df_comp_annual.columns:
                print(f"Warning: Column '{col}' not found for {competitor_name}. Ratios needing it will be NaN.")
                df_comp_annual[col] = np.nan
        
        if 'total_equity' not in df_comp_annual.columns or df_comp_annual['total_equity'].isnull().all():
             if ('total_assets' in df_comp_annual.columns and df_comp_annual['total_assets'].notna().any() and
                 'total_liabilities' in df_comp_annual.columns and df_comp_annual['total_liabilities'].notna().any()):
                df_comp_annual['total_equity'] = df_comp_annual['total_assets'] - df_comp_annual['total_liabilities']
                print(f"Calculated 'total_equity' for {competitor_name}.")
        
        df_comp_annual_sorted = df_comp_annual.sort_values(by='fiscal_year')
        for i, row_tuple in enumerate(df_comp_annual_sorted.itertuples(index=False)):
            row_data_dict = row_tuple._asdict()
            year = int(row_data_dict['fiscal_year'])
            all_calculated_ratios_list.extend([{'company_name':competitor_name,'fiscal_year':year,**r} for r in calculate_profitability_ratios(row_data_dict)])
            all_calculated_ratios_list.extend([{'company_name':competitor_name,'fiscal_year':year,**r} for r in calculate_liquidity_ratios(row_data_dict)])
            all_calculated_ratios_list.extend([{'company_name':competitor_name,'fiscal_year':year,**r} for r in calculate_solvency_ratios(row_data_dict)])
            all_calculated_ratios_list.extend([{'company_name':competitor_name,'fiscal_year':year,**r} for r in calculate_efficiency_ratios(row_data_dict)])
            if i > 0:
                prev_row_data_dict = df_comp_annual_sorted.iloc[i-1].to_dict()
                all_calculated_ratios_list.extend([{'company_name':competitor_name,'fiscal_year':year,**r} for r in calculate_growth_ratios(row_data_dict, prev_row_data_dict)])
    
    # Store Calculated Ratios in the Database
    if all_calculated_ratios_list:
        df_ratios_to_insert = pd.DataFrame(all_calculated_ratios_list)
        df_ratios_to_insert.dropna(subset=['ratio_value'], inplace=True)

        if not df_ratios_to_insert.empty:
            print(f"\nInserting/Updating {len(df_ratios_to_insert)} calculated ratios into database...")
            for _, ratio_row in df_ratios_to_insert.iterrows():
                ratio_val = ratio_row['ratio_value']
                if isinstance(ratio_val, (np.floating, np.integer)): ratio_val = float(ratio_val)
                if pd.isna(ratio_val): ratio_val = None
                
                cur.execute("""
                    INSERT INTO company_financial_ratios (company_name, fiscal_year, ratio_category, ratio_name, ratio_value)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (company_name, fiscal_year, ratio_name) DO UPDATE SET
                        ratio_value = EXCLUDED.ratio_value,
                        ratio_category = EXCLUDED.ratio_category; 
                """, (str(ratio_row['company_name']), int(ratio_row['fiscal_year']), 
                      str(ratio_row['ratio_category']), str(ratio_row['ratio_name']), ratio_val))
            conn.commit()
            print("Ratios successfully inserted/updated.")
        else:
            print("No valid non-NaN ratios were calculated to insert.")
    else:
        print("No ratios were calculated.")

except (Exception, psycopg2.Error) as error:
    print(f"Error during ratio calculation or DB operation: {error}")
    import traceback
    traceback.print_exc()
    if conn: conn.rollback()
finally:
    if conn:
        cur.close()
        conn.close()
        print("\nPostgreSQL connection closed.")

print("\nBenchmark Ratio Calculation Process Completed.")