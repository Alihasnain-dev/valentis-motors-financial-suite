import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
import matplotlib.pyplot as plt
import numpy as np

# Configuration
performance_data_path = "department_performance_scores.csv"

# Weights for the objective function components
WEIGHT_ROI = 0.50
WEIGHT_EFFICIENCY = 0.20
WEIGHT_STRATEGIC = 0.30

# Fixed total budgets for each scenario
BUDGETS_BY_SCENARIO = {
    "Baseline": 17000000.00,
    "TariffScenarioExtended": 14800000.00
}

# Main function to run optimization for a given scenario
def run_optimization(scenario_name, total_available_budget, df_perf):
    print(f"\nRunning Optimization for Scenario: {scenario_name}")
    print(f"Total Available Budget: {total_available_budget:,.2f}")

    # Prepare Department Data and Calculate Value Scores
    departments_data = []
    for index, row in df_perf.iterrows():
        value_factor = (WEIGHT_ROI * row['mock_roi_score'] +
                        WEIGHT_EFFICIENCY * row['mock_operational_efficiency_score'] +
                        WEIGHT_STRATEGIC * row['strategic_importance_score'])
        departments_data.append({
            'name': row['department_name'], 'value_factor': value_factor,
            'requested': row['requested_budget_next_year'],
            'min_req': row['min_required_budget_next_year'],
            'max_plaus': row['max_plausible_budget_next_year']
        })

    # Define the Optimization Problem with PuLP
    prob = LpProblem(f"BudgetAllocation_{scenario_name}", LpMaximize)
    
    # Define Decision Variables for each department's budget
    for dept_info in departments_data:
        dept_name = dept_info['name']
        safe_dept_name = "".join(filter(str.isalnum, dept_name))
        min_budget = dept_info['min_req'] if pd.notna(dept_info['min_req']) else 0
        max_budget = dept_info['max_plaus'] if pd.notna(dept_info['max_plaus']) else dept_info['requested'] * 1.5
        if min_budget > max_budget: max_budget = min_budget
        dept_info['variable'] = LpVariable(f"budget_{safe_dept_name}", lowBound=min_budget, upBound=max_budget)

    # Define Objective Function and Constraints
    prob += lpSum([dept['variable'] * dept['value_factor'] for dept in departments_data]), "TotalWeightedValueContribution"
    prob += lpSum([dept['variable'] for dept in departments_data]) <= total_available_budget, "TotalBudgetConstraint"

    # Solve the optimization problem
    prob.solve()

    # Process and return results
    if LpStatus[prob.status] == 'Optimal':
        results_list = []
        for dept in departments_data:
            results_list.append({
                'Scenario': scenario_name,
                'Department': dept['name'],
                'Requested_Budget': dept['requested'],
                'Min_Required': dept['min_req'],
                'Max_Plausible': dept['max_plaus'],
                'Optimized_Allocation': dept['variable'].value()
            })
        df_results = pd.DataFrame(results_list)
        print(f"Optimization successful for {scenario_name}.")
        return df_results
    else:
        print(f"Optimization FAILED for {scenario_name}. Status: {LpStatus[prob.status]}")
        return None

# Main Script Execution
print("Starting Budget Optimization Process for All Scenarios...")

try:
    df_perf_scores = pd.read_csv(performance_data_path)
    print(f"Loaded department performance data. Shape: {df_perf_scores.shape}")
except Exception as e:
    print(f"FATAL ERROR: Could not load performance data file '{performance_data_path}'. {e}")
    exit()

# Run optimization for all defined scenarios
all_results = []
for scenario, budget in BUDGETS_BY_SCENARIO.items():
    result_df = run_optimization(scenario, budget, df_perf_scores)
    if result_df is not None:
        all_results.append(result_df)

# Combine and Save Results
if all_results:
    df_combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save the combined results to a single CSV
    combined_output_path = "optimized_budgets_2025_ALL_SCENARIOS.csv"
    df_combined_results.to_csv(combined_output_path, index=False)
    print(f"\nSuccessfully saved combined optimization results to: {combined_output_path}")

    # Generate Comparison Plot
    print("Generating comparison plot...")
    
    # Pivot the data for plotting
    df_pivot = df_combined_results.pivot(index='Department', columns='Scenario', values='Optimized_Allocation')
    
    # Ensure consistent plotting order
    if "Baseline" in df_pivot.columns and "TariffScenarioExtended" in df_pivot.columns:
        df_pivot = df_pivot[["Baseline", "TariffScenarioExtended"]]
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set positions and width for the bars
    positions = np.arange(len(df_pivot.index))
    width = 0.35

    # Plot the bars for each scenario
    bar1 = ax.bar(positions - width/2, df_pivot['Baseline'], width, label='Baseline Budget', color='royalblue')
    bar2 = ax.bar(positions + width/2, df_pivot['TariffScenarioExtended'], width, label='Tariff Scenario Budget', color='indianred')

    # Add labels, title, and ticks
    ax.set_ylabel('Allocated Budget ($)')
    ax.set_title('Optimized Budget Allocation: Baseline vs. Tariff Scenario (2025)')
    ax.set_xticks(positions)
    ax.set_xticklabels(df_pivot.index, rotation=45, ha="right")
    ax.legend()
    
    # Format Y-axis for readability
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    fig.tight_layout()

    # Save and show the plot
    comparison_plot_path = "plot_budget_optimization_comparison.png"
    plt.savefig(comparison_plot_path)
    print(f"Successfully saved comparison plot to: {comparison_plot_path}")
    plt.show()

else:
    print("\nNo optimal solutions found. Cannot generate results or plots.")

print("\nBudget Optimization Process Completed.")