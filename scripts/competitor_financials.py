import yfinance as yf
import pandas as pd

# Configuration
tickers = {
    'Ford': 'F',
    'GM':   'GM',
    'Tesla':'TSLA'
}
years = [2021, 2022, 2023, 2024]

# Define desired columns from the income statement and their corresponding yfinance names
income_cols = {
    'Total Revenue':     'Total Revenue',
    'Cost of Revenue':   'Cost Of Revenue',
    'Gross Profit':      'Gross Profit',
    'R&D Expense':       'Research Development',
    'SG&A Expense':      'Selling General Administrative',
    'Operating Income':  'Operating Income',
    'Interest Expense':  'Interest Expense',
    'Income Tax Expense':'Income Tax Expense',
    'Net Income':        'Net Income'
}

# Define desired columns from the balance sheet and their corresponding yfinance names
balance_cols = {
    'Total Current Assets':'Total Current Assets',
    'Total Assets':         'Total Assets',
    'Total Current Liab':   'Total Current Liabilities',
    'Total Liabilities':    'Total Liab',
    'Total Equity':         'Total Stockholder Equity',
    'Cash & Equivalents':   'Cash Cash Equivalents',
    'Inventory':            'Inventory',
    'AR':                   'Net Receivables',
    'AP':                   'Accounts Payable',
    'Long-Term Debt':       'Long Term Debt'
}

# Function to pull and structure annual financial data for a given ticker
def fetch_annual_statements(ticker: str):
    tk = yf.Ticker(ticker)
    
    # Fetch annual Income Statement and Balance Sheet
    inc = tk.financials
    bal = tk.balancesheet
    
    # Transpose dataframes to have years as rows and financial items as columns
    inc = inc.T
    bal = bal.T
    
    # Filter data to the specified years
    inc['Year'] = inc.index.year
    bal['Year'] = bal.index.year
    inc = inc[inc['Year'].isin(years)]
    bal = bal[bal['Year'].isin(years)]
    return inc, bal

# Main Script Execution
if __name__ == "__main__":
    for name, sym in tickers.items():
        print(f"Fetching data for {name} ({sym})...")
        inc, bal = fetch_annual_statements(sym)

        # Process and save the Income Statement
        inc_out = pd.DataFrame({ 
            key: inc.get(col) 
            for key, col in income_cols.items() 
            if col in inc.columns
        })
        inc_out.insert(0, 'Year', inc['Year'].values)
        csv_inc = f"{name}_Income_Statement_2021_2024.csv"
        inc_out.to_csv(csv_inc, index=False)
        print(f" • Saved {csv_inc}")

        # Process and save the Balance Sheet
        bal_out = pd.DataFrame({
            key: bal.get(col)
            for key, col in balance_cols.items()
            if col in bal.columns
        })
        bal_out.insert(0, 'Year', bal['Year'].values)
        csv_bal = f"{name}_Balance_Sheet_2021_2024.csv"
        bal_out.to_csv(csv_bal, index=False)
        print(f" • Saved {csv_bal}")

    print("\nAll files written.")