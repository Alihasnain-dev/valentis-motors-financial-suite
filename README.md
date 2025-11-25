# Valentis Motors: Integrated Strategic Financial Planning & Analysis Suite

## Overview

This project is a comprehensive financial analysis and forecasting suite built for "Valentis Motors," a mock global automotive manufacturer producing both traditional (ICE) and electric (EV) vehicles. The suite provides end-to-end analytical capabilities, from data ingestion and preparation to multi-component forecasting, scenario analysis, and strategic budget optimization.

This portfolio piece demonstrates advanced skills in data engineering, time-series forecasting, financial modeling, and optimization to drive strategic business decisions.

## Module 1: Predictive Cash Flow Forecasting & Scenario Analysis

This module is the core of the suite, designed to move beyond basic trend extrapolation to **Causal AI Modeling**. It predicts Operating Revenue, Cost of Goods Sold (COGS), and Operating Expenses (Opex) by modeling their statistical relationships with external macroeconomic drivers.

### Key Methodological Innovations

1.  **Hybrid Training Strategy (Synthetic + Real Data)**
    *   **The Problem:** Real-world historical data (2021–Present) was too short to capture long-term economic cycles and lacked the variance needed to train the model on cost sensitivities (e.g., how COGS reacts to high inflation).
    *   **The Solution:** Implemented a **Synthetic History Generator** (`build_synthetic_history.py`) that creates 20 years of proxy data (2000–2021) using real-world benchmarks (e.g., US Total Vehicle Sales, CPI).
    *   **The "Intellectual" Leap:** Instead of random noise, the synthetic generation uses **Causal Logic**. Synthetic COGS was mathematically derived from the `EV_Component_Cost_Index` (CPI proxy). This forced the Prophet model to "learn" a positive correlation between input costs and COGS, correcting a previous flaw where the model incorrectly assumed higher costs improved margins.

2.  **Driver-Based Scenario Modeling (Input vs. Output Manipulation)**
    *   **Shift:** Moved from "Hardcoded Heuristics" to "Exogenous Shock Modeling."
    *   **Approach:** Instead of hardcoding the results (e.g., "Tariff Revenue = 80% of Baseline"), I implemented `generate_tariff_indicators.py` to shock the **inputs** instead.
        *   *Tariff Scenario Definition:* A 25% shock to the `EV_Component_Cost_Index` and a 15% drop in `Consumer_Confidence_Index`.
    *   **Result:** The model **endogenously** predicts margin compression and revenue suppression based on learned elasticities, providing a scientifically robust "What-If" analysis without manual overrides.

3.  **Data Quality "Sanity Patching"**
    *   **Problem:** Historical Opex data contained a "Cliff" (values <1% of revenue) due to source data corruption.
    *   **Fix:** Implemented an automated pre-processing rule in the ETL stage. If `Opex < 1% Revenue`, the script imputes a value of `15% Revenue`. This smoothed the forecast and removed artificial volatility.

### Technical Implementation

*   **Model:** Facebook Prophet with Multiplicative Seasonality and Regressors.
*   **Regressors:**
    *   `Interest_Rate_Proxy` (Negative correlation to Revenue validated).
    *   `Consumer_Confidence_Index` (Positive correlation to Revenue validated).
    *   `EV_Component_Cost_Index` (Positive correlation to COGS validated via Synthetic training).
*   **Validation:**
    *   **Method:** Blind Backtest (Training on pre-2023 data, testing against 2023–2024 "Truth").
    *   **Performance:** Achieved a **MAPE (Mean Absolute Percentage Error) of 9.66%**, categorized as "High-Fidelity."

## Module 2: Intelligent Budget Allocation Optimizer

This module moves beyond reporting to **Optimization and Decision Support**, demonstrating how human-AI collaboration can solve complex resource allocation problems.

### Core Functionality
*   **Objective:** Maximize a defined strategic objective (e.g., Weighted ROI, Operational Efficiency) subject to strict financial constraints.
*   **Input Data:** Mock departmental budget requests and historical performance metrics (ROI on marketing spend, operational efficiency scores).
*   **Optimization Model:** Built using Python's **PuLP** library (Linear Programming).
    *   **Decision Variables:** Allocation amounts per department.
    *   **Constraints:** Total Budget Cap (Baseline vs. Tariff scenarios), Minimum Departmental Funding levels.
    *   **Objective Function:** Maximize $\sum (Allocation_i \times ROI_i \times StrategicWeight_i)$.

### Strategic Value
*   Demonstrates higher-level analytical thinking by shifting focus from "Where did the money go?" to "Where *should* the money go?"
*   Provides actionable recommendations for reallocating funds during the Tariff Scenario (e.g., cutting low-ROI areas to preserve R&D).

## Module 3: AI-Enhanced Competitor Financial Benchmarking Tool

This module integrates external market intelligence to benchmark Valentis Motors against real-world industry leaders (e.g., Ford, GM, Tesla).

### Key Components
1.  **Automated Data Collection:**
    *   Python scripts scrape or fetch key public financial data (Revenue, Net Income, R&D Spend) for competitor companies.
    *   Data is stored in the SQL database for structured analysis.
2.  **Financial Ratio Analysis:**
    *   Calculates standardized metrics (Net Profit Margin, R&D % of Revenue, Current Ratio) to allow for apples-to-apples comparisons.
3.  **AI-Driven Qualitative Insights:**
    *   Uses an LLM API to ingest and summarize key findings from competitors' annual reports or news releases.
    *   Combines quantitative charts (Power BI) with qualitative AI-generated summaries to provide a holistic view of the competitive landscape.

### Strategic Value
*   Shows the ability to integrate external unstructured data (text) with structured financial metrics.
*   Enables "Peer Group Analysis" to contextualize Valentis Motors' performance against market standards.

## Tech Stack

*   **Language:** Python
*   **Data Manipulation & Analysis:** Pandas, NumPy
*   **Database:** PostgreSQL (with `psycopg2` for connectivity)
*   **Forecasting:** Prophet (formerly Facebook Prophet)
*   **Optimization:** PuLP
*   **Data Visualization:** Matplotlib
*   **Development Environment:** VS Code, Jupyter Notebooks

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/valentis-motors-financial-suite.git
    cd valentis-motors-financial-suite
    ```

2.  **Set up the environment:**
    *   Create and activate a Python virtual environment:
      ```bash
      python -m venv .venv
      .\.venv\Scripts\Activate.ps1
      ```
    *   Install the required libraries:
      ```bash
      pip install -r requirements.txt
      ```

3.  **Setup the Database:**
    *   Ensure you have PostgreSQL running.
    *   Create a database named `valentis_motors_db`.
    *   Set the environment variable `VALENTIS_DB_PASSWORD` to your PostgreSQL password.
    *   Run the data loading script (you will need the `Valentis_Motors_Data.xlsx` file, which is not included in this repo):
      ```bash
      python scripts/load_excel_to_postgres.py
      ```

4.  **Run the analysis scripts in order:**
    ```bash
    # (Optional) Fetch competitor data
    python scripts/competitor_financials.py
    python scripts/load_competitor_data.py
    
    # Main analysis workflow
    python scripts/cash_flow_forecasting_prep.py
    # (Optional) Generate long-run synthetic history for training
    python scripts/build_synthetic_history.py
    # Forecast both scenarios (Baselines + Tariff shocks)
    python scripts/run_scenario_forecast.py baseline
    python scripts/run_scenario_forecast.py tariff
    python scripts/calculate_benchmark_ratios.py
    python scripts/budget_optimizer.py
    # Build final summaries and comparison plot from the model outputs
    python scripts/build_and_compare_scenarios.py
    ```

## Showcasing Results

Here are some of the outputs generated by the analysis suite.

### Baseline vs. Tariff Scenario Revenue Forecast

This chart visualizes the projected impact of trade tariffs on Valentis Motors' operating revenue.
<img width="1500" height="800" alt="image" src="https://github.com/user-attachments/assets/0e71840b-8f44-4418-a18d-7ddaf5763587" />



### Optimized Budget Allocation
<img width="1400" height="800" alt="image" src="https://github.com/user-attachments/assets/c6b66220-5a3c-4e79-a5c1-9b8660b82c0e" />

This demonstrates how the budget optimizer reallocates funds from a baseline budget to a more constrained tariff scenario budget, prioritizing departments with the highest strategic value.
