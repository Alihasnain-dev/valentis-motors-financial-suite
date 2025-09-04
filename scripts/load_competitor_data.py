import pandas as pd
import psycopg2
from psycopg2 import sql
import os
import re
import numpy as np

# Database Connection Details
DB_NAME = "valentis_motors_db"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("VALENTIS_DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"

# Configuration for Competitors and their data files
COMPETITORS_CONFIG = [
    {
        "name": "Tesla, Inc.", "ticker": "TSLA",
        "income_statement_csv": "tesla_income_statement.csv",
        "balance_sheet_csv": "tesla_balance_sheet.csv"
    },
    {
        "name": "Ford Motor Company", "ticker": "F",
        "income_statement_csv": "ford_income_statement.csv",
        "balance_sheet_csv": "ford_balance_sheet.csv"
    },
    {
        "name": "General Motors Company", "ticker": "GM",
        "income_statement_csv": "gm_income_statement.csv",
        "balance_sheet_csv": "gm_balance_sheet.csv"
    }
]

# Defines which columns (post-cleaning) should be treated as numeric financials
EXPECTED_NUMERIC_COLS = [
    'total_revenue', 'cost_of_revenue', 'gross_profit', 'operating_income',
    'interest_expense', 'net_income', 'total_assets', 'inventory',
    'ap', 'accounts_payable', 'long_term_debt'
]

# Maps cleaned CSV column names to final SQL table column names
INCOME_STATEMENT_COL_MAP = {
    'total_revenue': 'total_revenue', 'cost_of_revenue': 'cost_of_revenue',
    'gross_profit': 'gross_profit', 'operating_income': 'operating_income',
    'interest_expense': 'interest_expense', 'net_income': 'net_income'
}
BALANCE_SHEET_COL_MAP = {
    'total_assets': 'total_assets', 'inventory': 'inventory',
    'ap': 'accounts_payable',
    'long_term_debt': 'long_term_debt'
}

# Helper function to standardize financial column names
def clean_financial_col_name(col_name):
    name = str(col_name).strip().lower()
    name = name.replace(' ', '_').replace('-', '_').replace('.', '')
    name = re.sub(r'_+', '_', name)
    return name

# Helper function to safely convert values to float
def convert_to_float_or_none(value):
    if pd.isna(value): return None
    if isinstance(value, (int, float)): return float(value)
    try:
        cleaned_value = str(value).replace(',', '').strip()
        return float(cleaned_value) if cleaned_value else None
    except (ValueError, TypeError): return None

print("Starting Competitor Data Loading Process...")
conn = None
try:
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()

    for competitor_info in COMPETITORS_CONFIG:
        competitor_name = competitor_info["name"]
        ticker = competitor_info["ticker"]
        is_csv_path = competitor_info["income_statement_csv"]
        bs_csv_path = competitor_info["balance_sheet_csv"]
        print(f"\nProcessing competitor: {competitor_name}")

        # Insert or update competitor in the lookup table and get its ID
        cur.execute("""
            INSERT INTO competitors (competitor_name, ticker_symbol) VALUES (%s, %s)
            ON CONFLICT (competitor_name) DO UPDATE SET ticker_symbol = EXCLUDED.ticker_symbol
            RETURNING competitor_id;
        """, (competitor_name, ticker))
        competitor_id_result = cur.fetchone()
        competitor_id = competitor_id_result[0] if competitor_id_result else None
        if not competitor_id:
             cur.execute("SELECT competitor_id FROM competitors WHERE competitor_name = %s;", (competitor_name,))
             competitor_id_result_fallback = cur.fetchone()
             competitor_id = competitor_id_result_fallback[0] if competitor_id_result_fallback else None
        if not competitor_id:
            print(f"CRITICAL ERROR: Could not get or create competitor_id for {competitor_name}. Skipping.")
            if conn: conn.rollback()
            continue
        conn.commit()
        print(f"Using competitor_id: {competitor_id} for {competitor_name}")

        df_is_load, df_bs_load = pd.DataFrame(), pd.DataFrame()

        # Load and process Income Statement data
        try:
            df_is_raw = pd.read_csv(is_csv_path, dtype=str)
            df_is_raw.columns = [clean_financial_col_name(col) for col in df_is_raw.columns]
            if 'year' in df_is_raw.columns:
                df_is_raw.rename(columns={'year': 'fiscal_year'}, inplace=True)
            
            if 'fiscal_year' in df_is_raw.columns:
                df_is_raw['fiscal_year'] = pd.to_numeric(df_is_raw['fiscal_year'], errors='coerce').astype('Int64')
            else:
                raise ValueError("Column 'Year' (for fiscal_year) not found in Income Statement CSV.")

            for col in df_is_raw.columns:
                if col in EXPECTED_NUMERIC_COLS and col != 'fiscal_year':
                    df_is_raw[col] = df_is_raw[col].apply(convert_to_float_or_none)
            
            cols_to_select_is = ['fiscal_year'] + [k for k,v in INCOME_STATEMENT_COL_MAP.items() if k in df_is_raw.columns]
            for sql_name in INCOME_STATEMENT_COL_MAP.values():
                if sql_name in df_is_raw.columns and sql_name not in cols_to_select_is:
                    cols_to_select_is.append(sql_name)
            
            df_is_load = df_is_raw[list(set(cols_to_select_is))].copy()
            df_is_load.rename(columns=INCOME_STATEMENT_COL_MAP, inplace=True)
            print(f"Loaded and cleaned Income Statement for {competitor_name}. Shape: {df_is_load.shape}.")

        except FileNotFoundError: print(f"ERROR: IS CSV not found for {competitor_name} at {is_csv_path}. Skipping."); continue
        except ValueError as ve: print(f"ValueError processing IS CSV for {competitor_name}: {ve}"); continue
        except Exception as e: print(f"Error processing IS CSV for {competitor_name}: {e}"); continue
            
        # Load and process Balance Sheet data
        try:
            df_bs_raw = pd.read_csv(bs_csv_path, dtype=str)
            df_bs_raw.columns = [clean_financial_col_name(col) for col in df_bs_raw.columns]
            if 'year' in df_bs_raw.columns:
                df_bs_raw.rename(columns={'year': 'fiscal_year'}, inplace=True)

            if 'fiscal_year' in df_bs_raw.columns:
                df_bs_raw['fiscal_year'] = pd.to_numeric(df_bs_raw['fiscal_year'], errors='coerce').astype('Int64')
            else:
                raise ValueError("Column 'Year' (for fiscal_year) not found in Balance Sheet CSV.")

            for col in df_bs_raw.columns:
                if col in EXPECTED_NUMERIC_COLS and col != 'fiscal_year':
                    df_bs_raw[col] = df_bs_raw[col].apply(convert_to_float_or_none)
            
            cols_to_select_bs = ['fiscal_year'] + [k for k,v in BALANCE_SHEET_COL_MAP.items() if k in df_bs_raw.columns]
            for sql_name in BALANCE_SHEET_COL_MAP.values():
                if sql_name in df_bs_raw.columns and sql_name not in cols_to_select_bs:
                     cols_to_select_bs.append(sql_name)

            df_bs_load = df_bs_raw[list(set(cols_to_select_bs))].copy()
            df_bs_load.rename(columns=BALANCE_SHEET_COL_MAP, inplace=True)
            print(f"Loaded and cleaned Balance Sheet for {competitor_name}. Shape: {df_bs_load.shape}.")

        except FileNotFoundError: print(f"ERROR: BS CSV not found for {competitor_name} at {bs_csv_path}. Skipping."); continue
        except ValueError as ve: print(f"ValueError processing BS CSV for {competitor_name}: {ve}"); continue
        except Exception as e: print(f"Error processing BS CSV for {competitor_name}: {e}"); continue
        
        # Merge financial statement dataframes
        if not df_is_load.empty and 'fiscal_year' in df_is_load.columns and \
           not df_bs_load.empty and 'fiscal_year' in df_bs_load.columns:
            df_merged_competitor = pd.merge(df_is_load, df_bs_load, on='fiscal_year', how='outer')
        elif not df_is_load.empty and 'fiscal_year' in df_is_load.columns: df_merged_competitor = df_is_load
        elif not df_bs_load.empty and 'fiscal_year' in df_bs_load.columns: df_merged_competitor = df_bs_load
        else: print(f"No valid financial data (with fiscal_year) for {competitor_name}. Skipping insert."); continue
        
        df_merged_competitor['competitor_id'] = competitor_id

        # Insert/Update each year of financial data for the competitor
        sql_table_columns_db = [
            'competitor_id', 'fiscal_year', 'total_revenue', 'cost_of_revenue', 'gross_profit',
            'operating_income', 'interest_expense', 'net_income', 'total_assets',
            'inventory', 'accounts_payable', 'long_term_debt'
        ]
        for _, row in df_merged_competitor.iterrows():
            insert_data_dict = {}
            for sql_col in sql_table_columns_db:
                value = row.get(sql_col)
                if pd.isna(value): insert_data_dict[sql_col] = None
                elif isinstance(value, np.integer): insert_data_dict[sql_col] = int(value)
                elif isinstance(value, np.floating): insert_data_dict[sql_col] = float(value)
                else: insert_data_dict[sql_col] = value
            if insert_data_dict.get('fiscal_year') is not None:
                try: insert_data_dict['fiscal_year'] = int(insert_data_dict['fiscal_year'])
                except ValueError: print(f"Warning: Could not convert fiscal_year '{insert_data_dict['fiscal_year']}' to int for {competitor_name}. Row skipped."); continue
            else: print(f"Warning: fiscal_year is missing for a row in {competitor_name}. Skipping row."); continue

            update_assignments_sql = []
            cols_for_insert_sql_ident = []
            values_for_insert_sql = []
            for col_name, val_to_insert in insert_data_dict.items():
                if col_name in sql_table_columns_db:
                    cols_for_insert_sql_ident.append(sql.Identifier(col_name))
                    values_for_insert_sql.append(val_to_insert)
                    if col_name not in ['competitor_id', 'fiscal_year']:
                        update_assignments_sql.append(sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col_name), sql.Identifier(col_name)))
            if not cols_for_insert_sql_ident: print(f"No valid columns to insert for {competitor_name}, year {row.get('fiscal_year')}"); continue
            
            insert_cols_str_sql = sql.SQL(', ').join(cols_for_insert_sql_ident)
            placeholders_sql = sql.SQL(', ').join(sql.Placeholder() * len(values_for_insert_sql))
            conflict_action_sql = sql.SQL("DO UPDATE SET ") + sql.SQL(', ').join(update_assignments_sql) if update_assignments_sql else sql.SQL("DO NOTHING")
            query = sql.SQL("INSERT INTO competitor_annual_financials ({}) VALUES ({}) ON CONFLICT (competitor_id, fiscal_year) {}").format(
                insert_cols_str_sql, placeholders_sql, conflict_action_sql)
            try: cur.execute(query, values_for_insert_sql)
            except Exception as db_err: print(f"DB ERROR for {competitor_name}, year {row.get('fiscal_year')}: {db_err}\nData: {insert_data_dict}"); conn.rollback(); break 
        conn.commit()
        print(f"Data for {competitor_name} loaded/updated.")

except (Exception, psycopg2.Error) as error: print(f"Error: {error}");
finally:
    if conn: cur.close(); conn.close(); print("\nPostgreSQL connection closed.")
print("\nCompetitor Data Loading Process Completed.")