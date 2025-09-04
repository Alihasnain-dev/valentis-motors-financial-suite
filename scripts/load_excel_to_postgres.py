import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
import re
import numpy as np
import os

# Database Connection Details
DB_NAME = "valentis_motors_db"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("VALENTIS_DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"

# Excel File Path
excel_file_path = "Valentis_Motors_Data.xlsx"

# SQLAlchemy Engine
engine_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(engine_str)
print(f"Successfully created SQLAlchemy engine for database: {DB_NAME}")

# Dictionaries for Foreign Key Mapping
department_map = {}
vehicle_model_map = {}
region_map = {}
manufacturing_plant_map = {}
supplier_map = {}
rd_center_map = {}

# Helper to clean column names for standardization
def clean_col_name(column_name):
    name = str(column_name).strip()
    name = re.sub(r'\s+|\W+', '_', name) # Replace spaces and non-alphanumeric with _
    name = re.sub(r'_+', '_', name)      # Replace multiple underscores with single
    name = name.strip('_')              # Strip leading/trailing underscores
    return name.lower()

print("Starting ETL Process...")

# Step 1: Load Independent Lookup Tables & Populate ID Maps
def load_lookup_tables_and_maps():
    global department_map, vehicle_model_map, region_map, manufacturing_plant_map, supplier_map, rd_center_map
    conn = None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()

        # Process Regions
        print("\nProcessing Regions")
        all_region_names = set()
        sheets_with_region = ["ASP_Data", "Sales_Production", "Order_Backlog"]
        for sheet_name in sheets_with_region:
            try:
                df_reg_source = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                if 'Region' in df_reg_source.columns:
                    all_region_names.update(df_reg_source['Region'].dropna().unique())
            except Exception as e: print(f"Warning: Could not read 'Region' from {sheet_name}: {e}")
        for name in sorted(list(all_region_names)):
            cur.execute("INSERT INTO regions (region_name) VALUES (%s) ON CONFLICT (region_name) DO UPDATE SET region_name = EXCLUDED.region_name RETURNING region_id, region_name", (name,))
            result = cur.fetchone()
            if result: region_map[result[1]] = result[0]
        conn.commit()
        print(f"Regions loaded. Map: {region_map}")

        # Process Departments
        print("\nProcessing Departments")
        df_depts_excel = pd.read_excel(excel_file_path, sheet_name="Departments")
        for _, row in df_depts_excel.iterrows():
            dept_name, subunits_text = row['Department'], row['Subunits']
            if pd.notna(dept_name):
                cur.execute("INSERT INTO departments (department_name, subunits) VALUES (%s, %s) ON CONFLICT (department_name) DO UPDATE SET subunits = EXCLUDED.subunits RETURNING department_id, department_name", (dept_name, subunits_text))
                result = cur.fetchone()
                if result: department_map[result[1]] = result[0]
        conn.commit()
        print(f"Departments loaded. Map: {department_map}")

        # Process Vehicle Models & Variants
        print("\nProcessing Vehicle Models & Variants")
        df_vm_excel = pd.read_excel(excel_file_path, sheet_name="Vehicle_Models")
        for _, row in df_vm_excel.iterrows():
            model_name,v_type,segment,variant_name,desc = row['Model'],row['Type'],row['Segment'],row['Variant'],row['Description']
            cur.execute("INSERT INTO vehicle_models (model_name, vehicle_type, segment) VALUES (%s,%s,%s) ON CONFLICT (model_name) DO UPDATE SET vehicle_type=EXCLUDED.vehicle_type, segment=EXCLUDED.segment RETURNING model_id, model_name", (model_name,v_type,segment))
            model_res = cur.fetchone()
            curr_model_id = model_res[0]
            vehicle_model_map[model_res[1]] = curr_model_id
            if pd.notna(variant_name):
                 cur.execute("INSERT INTO vehicle_variants (model_id, variant_name, description) VALUES (%s,%s,%s) ON CONFLICT (model_id,variant_name) DO UPDATE SET description=EXCLUDED.description", (curr_model_id,variant_name,desc))
        conn.commit()
        print(f"Vehicle Models and Variants loaded. Model Map: {vehicle_model_map}")

        # Process Manufacturing Plants & Plant_Vehicle_Model_Production
        print("\nProcessing Manufacturing Plants")
        df_plants_excel = pd.read_excel(excel_file_path, sheet_name="Manufacturing_Plants")
        for _, row in df_plants_excel.iterrows():
            plant_name,location,models_prod_str = row['Plant'],row['Location'],row['Models_Produced']
            cur.execute("INSERT INTO manufacturing_plants (plant_name, location_country) VALUES (%s,%s) ON CONFLICT (plant_name) DO UPDATE SET location_country=EXCLUDED.location_country RETURNING plant_id, plant_name", (plant_name,location))
            plant_res = cur.fetchone()
            curr_plant_id = plant_res[0]
            manufacturing_plant_map[plant_res[1]] = curr_plant_id
            if pd.notna(models_prod_str):
                for model_p_name in [name.strip() for name in models_prod_str.split(';')]:
                    if model_p_id := vehicle_model_map.get(model_p_name):
                        cur.execute("INSERT INTO plant_vehicle_model_production (plant_id, model_id) VALUES (%s,%s) ON CONFLICT (plant_id,model_id) DO NOTHING", (curr_plant_id,model_p_id))
                    else: print(f"Warning: Model '{model_p_name}' for plant '{plant_name}' not found.")
        conn.commit()
        print(f"Manufacturing Plants and links loaded. Plant Map: {manufacturing_plant_map}")

        # Process Key Suppliers
        print("\nProcessing Key Suppliers")
        df_suppliers_excel = pd.read_excel(excel_file_path, sheet_name="Key_Suppliers")
        for _, row in df_suppliers_excel.iterrows():
            cur.execute("INSERT INTO key_suppliers (supplier_name, supply_type) VALUES (%s,%s) ON CONFLICT (supplier_name) DO UPDATE SET supply_type=EXCLUDED.supply_type RETURNING supplier_id, supplier_name", (row['Supplier'],row['Supply']))
            res = cur.fetchone()
            if res: supplier_map[res[1]] = res[0]
        conn.commit()
        print(f"Key Suppliers loaded. Map: {supplier_map}")

        # Process R&D Centers
        print("\nProcessing R&D Centers")
        df_rd_centers_excel = pd.read_excel(excel_file_path, sheet_name="R&D_Centers")
        for _, row in df_rd_centers_excel.iterrows():
            cur.execute("INSERT INTO rd_centers (center_name, country, focus_area) VALUES (%s,%s,%s) ON CONFLICT (center_name) DO UPDATE SET country=EXCLUDED.country, focus_area=EXCLUDED.focus_area RETURNING rd_center_id, center_name", (row['Center'],row['Country'],row['Focus']))
            res = cur.fetchone()
            if res: rd_center_map[res[1]] = res[0]
        conn.commit()
        print(f"R&D Centers loaded. Map: {rd_center_map}")

    except (Exception, psycopg2.Error) as error:
        print(f"GENERAL Error in load_lookup_tables_and_maps: {error}")
    finally:
        if conn: cur.close(); conn.close()

# Process Strategic Goals
def load_strategic_goals():
    print("\nProcessing Strategic Goals")
    try:
        df_goals_excel = pd.read_excel(excel_file_path, sheet_name="Strategic_Goals")
        def extract_year(desc_str):
            if pd.isna(desc_str): return None
            match = re.search(r'by\s+(\d{4})', str(desc_str))
            return int(match.group(1)) if match else None
        
        df_goals_load = pd.DataFrame()
        df_goals_load['goal_description'] = (df_goals_excel['Goal'].astype(str) + " - " + df_goals_excel['Description'].astype(str)).fillna('')
        df_goals_load['target_metric'] = df_goals_excel['Description'].astype(str).fillna('')
        df_goals_load['target_year'] = df_goals_excel['Description'].apply(extract_year)

        conn_goals = None
        try:
            conn_goals = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cur_goals = conn_goals.cursor()
            for _, row in df_goals_load.iterrows():
                db_target_year = row['target_year'] if pd.notna(row['target_year']) else None
                cur_goals.execute("""
                    INSERT INTO strategic_goals (goal_description, target_metric, target_year) VALUES (%s, %s, %s)
                    ON CONFLICT (goal_description) DO UPDATE SET target_metric = EXCLUDED.target_metric, target_year = EXCLUDED.target_year;
                """, (row['goal_description'], row['target_metric'], db_target_year))
            conn_goals.commit()
            print("Successfully processed Strategic_Goals data.")
        except Exception as e_db:
            if conn_goals: conn_goals.rollback()
            print(f"DB Error processing Strategic Goals: {e_db}")
        finally:
            if conn_goals: cur_goals.close(); conn_goals.close()
    except Exception as e_file:
        print(f"File/Pandas Error processing Strategic Goals: {e_file}")

# Main Execution Flow
load_lookup_tables_and_maps()

print("\nProcessing Company Profile")
try:
    df_profile_excel = pd.read_excel(excel_file_path, sheet_name="Company_Profile")
    def get_profile_value(df, attribute_name):
        series = df['Value'][df['Attribute'] == attribute_name]
        return series.values[0] if not series.empty and pd.notna(series.values[0]) else None

    company_name_val = get_profile_value(df_profile_excel, 'Company Name')
    if company_name_val is not None:
        profile_values = {
            "company_name": company_name_val,
            "founded_year": int(val) if pd.notna(val := get_profile_value(df_profile_excel, 'Founded')) else None,
            "headquarters": get_profile_value(df_profile_excel, 'Headquarters'),
            "global_regions": get_profile_value(df_profile_excel, 'Global Regions'),
            "brand_positioning": get_profile_value(df_profile_excel, 'Brand Positioning'),
            "slogan": get_profile_value(df_profile_excel, 'Slogan'),
            "employee_count": int(val) if pd.notna(val := get_profile_value(df_profile_excel, 'Employee Count')) else None
        }
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO company_profile (company_name, founded_year, headquarters, global_regions, brand_positioning, slogan, employee_count)
            VALUES (%(company_name)s, %(founded_year)s, %(headquarters)s, %(global_regions)s, %(brand_positioning)s, %(slogan)s, %(employee_count)s)
            ON CONFLICT (company_name) DO UPDATE SET
                founded_year = EXCLUDED.founded_year, headquarters = EXCLUDED.headquarters, global_regions = EXCLUDED.global_regions,
                brand_positioning = EXCLUDED.brand_positioning, slogan = EXCLUDED.slogan, employee_count = EXCLUDED.employee_count;
        """, profile_values)
        conn.commit()
        cur.close(); conn.close()
        print("Successfully loaded data into Company_Profile")
    else:
        print("Company Profile could not be loaded: 'Company Name' attribute not found or value is missing.")
except Exception as e:
    print(f"Error loading Company Profile: {e}")

load_strategic_goals()

# Processing Financial Statements
print("\nProcessing Financial Statements (Income, Balance, Cash Flow)")
financial_sheets_map = {
    "Income_Statement": "income_statement_data",
    "Balance_Sheet": "balance_sheet_data",
    "Cash_Flow": "cash_flow_data"
}
for sheet_name_excel, table_name_sql in financial_sheets_map.items():
    try:
        print(f"Processing sheet: {sheet_name_excel} for table: {table_name_sql}")
        df_fin = pd.read_excel(excel_file_path, sheet_name=sheet_name_excel)
        
        excel_col_to_cleaned_col = {col: clean_col_name(col) for col in df_fin.columns}
        df_fin.columns = [excel_col_to_cleaned_col[col] for col in df_fin.columns]

        sql_date_col_map = {
            "income_statement_data": "statement_date",
            "balance_sheet_data": "balance_date",
            "cash_flow_data": "flow_date"
        }
        current_sql_date_col = sql_date_col_map[table_name_sql]

        first_excel_cleaned_col_name = list(excel_col_to_cleaned_col.values())[0]
        if first_excel_cleaned_col_name != current_sql_date_col and first_excel_cleaned_col_name in df_fin.columns:
             df_fin.rename(columns={first_excel_cleaned_col_name: current_sql_date_col}, inplace=True)
        
        if current_sql_date_col in df_fin.columns:
            df_fin[current_sql_date_col] = pd.to_datetime(df_fin[current_sql_date_col]).dt.date
        else:
            print(f"WARNING: Date column '{current_sql_date_col}' not found in DataFrame for {table_name_sql} after cleaning. Original first col: {first_excel_cleaned_col_name}")

        conn_temp = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur_temp = conn_temp.cursor()
        cur_temp.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name_sql}';")
        sql_table_cols_from_db = [row[0] for row in cur_temp.fetchall()]
        cur_temp.close(); conn_temp.close()
        
        auto_pk_name_map = {
            "income_statement_data": "statement_id",
            "balance_sheet_data": "balance_sheet_id",
            "cash_flow_data": "cash_flow_id"
        }
        auto_pk_name = auto_pk_name_map.get(table_name_sql)

        df_load_cols = [col for col in df_fin.columns if col in sql_table_cols_from_db and col != auto_pk_name]
        
        if not df_load_cols:
            print(f"WARNING: No matching columns found to load for table {table_name_sql}. Check column names and SQL schema.")
            continue

        df_to_load_final = df_fin[df_load_cols].copy()

        for col in df_to_load_final.select_dtypes(include=np.number).columns:
            df_to_load_final[col] = pd.to_numeric(df_to_load_final[col], errors='coerce')
        df_to_load_final = df_to_load_final.where(pd.notnull(df_to_load_final), None)

        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        cols_for_insert_str = ", ".join([f"{c}" for c in df_to_load_final.columns])
        placeholders_str = ", ".join(["%s"] * len(df_to_load_final.columns))
        
        update_set_parts = [f"{col} = EXCLUDED.{col}" for col in df_to_load_final.columns if col != current_sql_date_col]
        update_set_clause = ", ".join(update_set_parts)
        
        if current_sql_date_col not in df_to_load_final.columns:
            print(f"CRITICAL ERROR: Date column '{current_sql_date_col}' for ON CONFLICT clause is not in the final set of columns to load for {table_name_sql}. Columns: {df_to_load_final.columns}")
            cur.close(); conn.close()
            continue

        insert_query = sql.SQL(f"INSERT INTO {table_name_sql} ({cols_for_insert_str}) VALUES ({placeholders_str}) ON CONFLICT ({current_sql_date_col}) DO UPDATE SET {update_set_clause};")
        
        print(f"Attempting to load {len(df_to_load_final)} rows into {table_name_sql} with columns: {df_to_load_final.columns.tolist()}")
        for _, row_data in df_to_load_final.iterrows():
            cur.execute(insert_query, tuple(row_data))
        conn.commit()
        cur.close(); conn.close()
        print(f"Successfully loaded data into {table_name_sql}")
    except Exception as e:
        print(f"Error loading {sheet_name_excel} into {table_name_sql}: {e}")

# Process Operating Expenses Data
print("\nProcessing Operating Expenses Data")
try:
    df_opex_excel = pd.read_excel(excel_file_path, sheet_name="Operating_Expenses")
    if department_map: 
        df_opex_load = pd.DataFrame({
            'expense_date': pd.to_datetime(df_opex_excel['Date']).dt.date,
            'department_id': df_opex_excel['Department'].map(department_map),
            'expense_amount': df_opex_excel['Expense']
        })
        df_opex_load.dropna(subset=['department_id', 'expense_date'], inplace=True)
        if not df_opex_load.empty:
            df_opex_load['department_id'] = df_opex_load['department_id'].astype(int)
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cur = conn.cursor()
            for _, row in df_opex_load.iterrows():
                cur.execute("""
                    INSERT INTO operating_expenses_data (expense_date, department_id, expense_amount) VALUES (%s, %s, %s)
                    ON CONFLICT (expense_date, department_id) DO UPDATE SET expense_amount = EXCLUDED.expense_amount;
                """, tuple(row))
            conn.commit()
            cur.close(); conn.close()
        print("Successfully loaded data into Operating_Expenses_Data")
    else:
        print("Skipping Operating_Expenses_Data: department_map is empty or no valid data after mapping.")
except Exception as e:
    print(f"Error loading Operating Expenses Data: {e}")

# Process Department Budgets
print("\nProcessing Department Budgets")
try:
    df_dept_bud_excel = pd.read_excel(excel_file_path, sheet_name="Dept_Budgets")
    if department_map:
        df_dept_bud_load = pd.DataFrame({
            'budget_year': df_dept_bud_excel['Year'],
            'department_id': df_dept_bud_excel['Department'].map(department_map),
            'budget_requested': df_dept_bud_excel['Budget_Requested'],
            'budget_actual': df_dept_bud_excel['Budget_Actual']
        })
        df_dept_bud_load.dropna(subset=['department_id', 'budget_year'], inplace=True)
        if not df_dept_bud_load.empty:
            df_dept_bud_load['department_id'] = df_dept_bud_load['department_id'].astype(int)
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cur = conn.cursor()
            for _, row in df_dept_bud_load.iterrows():
                cur.execute("""
                    INSERT INTO department_budgets (budget_year, department_id, budget_requested, budget_actual) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (budget_year, department_id) DO UPDATE SET budget_requested = EXCLUDED.budget_requested, budget_actual = EXCLUDED.budget_actual;
                """, tuple(row))
            conn.commit()
            cur.close(); conn.close()
        print("Successfully loaded data into Department_Budgets")
    else:
        print("Skipping Department_Budgets: department_map is empty or no valid data after mapping.")
except Exception as e:
    print(f"Error loading Department Budgets: {e}")

# Process Sales Production Data
print("\nProcessing Sales Production Data")
try:
    df_sales_prod_excel = pd.read_excel(excel_file_path, sheet_name="Sales_Production")
    if not region_map: print("CRITICAL WARNING: region_map is empty for Sales_Production_Data.")
    if not vehicle_model_map: print("CRITICAL WARNING: vehicle_model_map is empty for Sales_Production_Data.")

    expected_excel_cols = ['Date', 'Model', 'Region', 'Units_Sold', 'Units_Produced']
    missing_cols = [col for col in expected_excel_cols if col not in df_sales_prod_excel.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns in Sales_Production sheet: {missing_cols}")

    df_sales_prod_load = pd.DataFrame({
        'transaction_date': pd.to_datetime(df_sales_prod_excel['Date']).dt.date,
        'model_id': df_sales_prod_excel['Model'].map(vehicle_model_map),
        'region_id': df_sales_prod_excel['Region'].map(region_map),
        'units_sold': df_sales_prod_excel['Units_Sold'],
        'units_produced': df_sales_prod_excel['Units_Produced']
    })
    df_sales_prod_load.dropna(subset=['model_id', 'region_id', 'transaction_date'], inplace=True)
    if not df_sales_prod_load.empty:
        df_sales_prod_load['model_id'] = df_sales_prod_load['model_id'].astype(int)
        df_sales_prod_load['region_id'] = df_sales_prod_load['region_id'].astype(int)
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        insert_cols_sql = ['transaction_date', 'model_id', 'region_id', 'units_sold', 'units_produced']
        cols_str = ", ".join(insert_cols_sql)
        placeholders_str = ", ".join(["%s"] * len(insert_cols_sql))
        insert_query_sql = f"INSERT INTO sales_production_data ({cols_str}) VALUES ({placeholders_str});"
        for _, row in df_sales_prod_load.iterrows():
            values_tuple = tuple(row[col_name] for col_name in insert_cols_sql)
            cur.execute(insert_query_sql, values_tuple)
        conn.commit()
        cur.close(); conn.close()
        print("Successfully loaded data into Sales_Production_Data")
    else:
        print("Sales_Production_Data is empty after mapping/dropna or maps were empty.")
except KeyError as e: print(f"KeyError for Sales Production: {e}")
except Exception as e: print(f"General error for Sales Production: {e}")

# Process ASP Data
print("\nProcessing ASP Data")
try:
    df_asp_excel = pd.read_excel(excel_file_path, sheet_name="ASP_Data")
    if region_map and vehicle_model_map:
        df_asp_load = pd.DataFrame({
            'asp_year': df_asp_excel['Year'],
            'model_id': df_asp_excel['Model'].map(vehicle_model_map),
            'region_id': df_asp_excel['Region'].map(region_map),
            'average_selling_price': df_asp_excel['ASP']
        })
        df_asp_load.dropna(subset=['model_id', 'region_id', 'asp_year'], inplace=True)
        if not df_asp_load.empty:
            df_asp_load['model_id'] = df_asp_load['model_id'].astype(int)
            df_asp_load['region_id'] = df_asp_load['region_id'].astype(int)
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cur = conn.cursor()
            for _, row in df_asp_load.iterrows():
                cur.execute("""
                    INSERT INTO asp_data (asp_year, model_id, region_id, average_selling_price) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (asp_year, model_id, region_id) DO UPDATE SET average_selling_price = EXCLUDED.average_selling_price;
                """, tuple(row))
            conn.commit()
            cur.close(); conn.close()
        print("Successfully loaded data into ASP_Data")
    else:
        print("Skipping ASP_Data: maps empty or no valid data.")
except Exception as e:
    print(f"Error loading ASP Data: {e}")

# Process Order Backlog Data
print("\nProcessing Order Backlog Data")
try:
    df_backlog_excel = pd.read_excel(excel_file_path, sheet_name="Order_Backlog")
    if region_map and vehicle_model_map:
        df_backlog_load = pd.DataFrame({
            'backlog_date': pd.to_datetime(df_backlog_excel['Date']).dt.date,
            'model_id': df_backlog_excel['Model'].map(vehicle_model_map),
            'region_id': df_backlog_excel['Region'].map(region_map),
            'order_backlog_count': df_backlog_excel['Order_Backlog']
        })
        df_backlog_load.dropna(subset=['model_id', 'region_id', 'backlog_date'], inplace=True)
        if not df_backlog_load.empty:
            df_backlog_load['model_id'] = df_backlog_load['model_id'].astype(int)
            df_backlog_load['region_id'] = df_backlog_load['region_id'].astype(int)
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cur = conn.cursor()
            for _, row in df_backlog_load.iterrows():
                cur.execute("""
                    INSERT INTO order_backlog_data (backlog_date, model_id, region_id, order_backlog_count) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (backlog_date, model_id, region_id) DO UPDATE SET order_backlog_count = EXCLUDED.order_backlog_count;
                """, tuple(row))
            conn.commit()
            cur.close(); conn.close()
        print("Successfully loaded data into Order_Backlog_Data")
    else:
        print("Skipping Order_Backlog_Data: maps empty or no valid data.")
except Exception as e:
    print(f"Error loading Order Backlog Data: {e}")

# Process Plant Operational Data
print("\nProcessing Plant Operational Data (from Supply_Chain sheet)")
try:
    df_plant_op_excel = pd.read_excel(excel_file_path, sheet_name="Supply_Chain")
    if manufacturing_plant_map:
        df_plant_op_load = pd.DataFrame({
            'op_date': pd.to_datetime(df_plant_op_excel['Date']).dt.date,
            'plant_id': df_plant_op_excel['Plant'].map(manufacturing_plant_map),
            'battery_cost_per_kwh': df_plant_op_excel['Battery_Cost_per_kWh'],
            'engine_cost': df_plant_op_excel['Engine_Cost'],
            'capacity_utilization': df_plant_op_excel['Capacity_Utilization']
        })
        df_plant_op_load.dropna(subset=['plant_id', 'op_date'], inplace=True)
        if not df_plant_op_load.empty:
            df_plant_op_load['plant_id'] = df_plant_op_load['plant_id'].astype(int)
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            cur = conn.cursor()
            for _, row in df_plant_op_load.iterrows():
                cur.execute("""
                    INSERT INTO plant_operational_data (op_date, plant_id, battery_cost_per_kwh, engine_cost, capacity_utilization) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (op_date, plant_id) DO UPDATE SET
                        battery_cost_per_kwh = EXCLUDED.battery_cost_per_kwh, engine_cost = EXCLUDED.engine_cost, capacity_utilization = EXCLUDED.capacity_utilization;
                """, tuple(row))
            conn.commit()
            cur.close(); conn.close()
        print("Successfully loaded data into Plant_Operational_Data")
    else:
        print("Skipping Plant_Operational_Data: map empty or no valid data.")
except Exception as e:
    print(f"Error loading Plant Operational Data: {e}")

# Process R&D Projects
print("\nProcessing R&D Projects")
try:
    df_rd_excel = pd.read_excel(excel_file_path, sheet_name="R&D_Projects")
    if department_map:
        df_rd_load = pd.DataFrame({
            'rd_project_id': df_rd_excel['Project_ID'], 'project_name': df_rd_excel['Name'], 'project_year': df_rd_excel['Year'],
            'department_id': df_rd_excel['Lead_Department'].map(department_map),
            'budget_allocated': df_rd_excel['Budget'], 'actual_spend': df_rd_excel['Actual_Spend'],
            'timeline_start_date': pd.to_datetime(df_rd_excel['Timeline_Start'], errors='coerce').dt.date,
            'timeline_end_date': pd.to_datetime(df_rd_excel['Timeline_End'], errors='coerce').dt.date,
            'status': df_rd_excel['Status'], 'kpis': df_rd_excel['KPIs']
        })
        df_rd_load['department_id'] = df_rd_load['department_id'].apply(lambda x: int(x) if pd.notna(x) else None)
        df_rd_load['timeline_start_date'] = df_rd_load['timeline_start_date'].where(pd.notnull(df_rd_load['timeline_start_date']), None)
        df_rd_load['timeline_end_date'] = df_rd_load['timeline_end_date'].where(pd.notnull(df_rd_load['timeline_end_date']), None)
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        cur = conn.cursor()
        for _, row in df_rd_load.iterrows():
            cur.execute("""
                INSERT INTO rd_projects (rd_project_id, project_name, project_year, department_id, budget_allocated, actual_spend, timeline_start_date, timeline_end_date, status, kpis)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (rd_project_id) DO UPDATE SET
                    project_name=EXCLUDED.project_name, project_year=EXCLUDED.project_year, department_id=EXCLUDED.department_id, budget_allocated=EXCLUDED.budget_allocated,
                    actual_spend=EXCLUDED.actual_spend, timeline_start_date=EXCLUDED.timeline_start_date, timeline_end_date=EXCLUDED.timeline_end_date, status=EXCLUDED.status, kpis=EXCLUDED.kpis;
            """, tuple(row))
        conn.commit()
        cur.close(); conn.close()
        print("Successfully loaded data into RD_Projects")
    else:
        print("Skipping RD_Projects: department_map empty.")
except Exception as e:
    print(f"Error loading R&D Projects: {e}")

# Process Marketing Campaigns
print("\nProcessing Marketing Campaigns")
try:
    df_mc_excel = pd.read_excel(excel_file_path, sheet_name="Marketing_Campaigns")
    def clean_uplift(val):
        if pd.isna(val) or str(val).strip().upper() == 'N/A': return None
        try: return float(str(val).replace('%', '')) / 100
        except ValueError: return None
    df_mc_load = pd.DataFrame({
        'marketing_campaign_id': df_mc_excel['Campaign_ID'], 'campaign_name': df_mc_excel['Name'], 'campaign_year': df_mc_excel['Year'],
        'budget_allocated': df_mc_excel['Budget'], 'actual_spend': df_mc_excel['Actual_Spend'],
        'target_audience': df_mc_excel['Target_Audience'], 'channels': df_mc_excel['Channels'],
        'duration_months': df_mc_excel['Duration_Months'].apply(lambda x: int(x) if pd.notna(x) else None),
        'sales_uplift_estimate': df_mc_excel['Sales_Uplift_Est'].apply(clean_uplift)
    })
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    for _, row in df_mc_load.iterrows():
        cur.execute("""
            INSERT INTO marketing_campaigns (marketing_campaign_id, campaign_name, campaign_year, budget_allocated, actual_spend, target_audience, channels, duration_months, sales_uplift_estimate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (marketing_campaign_id) DO UPDATE SET
                campaign_name=EXCLUDED.campaign_name, campaign_year=EXCLUDED.campaign_year, budget_allocated=EXCLUDED.budget_allocated, actual_spend=EXCLUDED.actual_spend,
                target_audience=EXCLUDED.target_audience, channels=EXCLUDED.channels, duration_months=EXCLUDED.duration_months, sales_uplift_estimate=EXCLUDED.sales_uplift_estimate;
        """, tuple(row))
    conn.commit()
    cur.close(); conn.close()
    print("Successfully loaded data into Marketing_Campaigns")
except Exception as e:
    print(f"Error loading Marketing Campaigns: {e}")

print("\nETL Process Completed.")