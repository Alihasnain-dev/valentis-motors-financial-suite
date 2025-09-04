import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
input_data_path = "prepared_cash_flow_data.csv"
full_indicators_csv_path = "mock_economic_indicators.csv"

# List of targets to forecast
targets_to_forecast = [
    'inflow_operating_revenue',
    'outflow_cogs',
    'outflow_opex'
]

# Define economic indicators to use as regressors for all targets
common_regressor_columns = [
    'Economic_Activity_Index',
    'Consumer_Confidence_Index',
    'Interest_Rate_Proxy',
    'EV_Component_Cost_Index'
]

forecast_horizon_months = 24
test_set_months = 6

print("Starting Baseline Forecasts for Multiple Components...")

# Load and prepare master datasets
try:
    df_historical_master = pd.read_csv(input_data_path)
    date_col_name = None
    if 'statement_date' in df_historical_master.columns: date_col_name = 'statement_date'
    elif 'balance_date' in df_historical_master.columns: date_col_name = 'balance_date'
    elif df_historical_master.columns[0].lower() == 'date' or df_historical_master.columns[0].lower().endswith('_date'): date_col_name = df_historical_master.columns[0]
    elif 'Unnamed: 0' in df_historical_master.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_historical_master['Unnamed: 0'], errors='coerce')): date_col_name = 'Unnamed: 0'
    else: raise ValueError("Date column for Prophet ('ds') not found in prepared_cash_flow_data.csv.")
    
    df_historical_master.rename(columns={date_col_name: 'ds'}, inplace=True)
    df_historical_master['ds'] = pd.to_datetime(df_historical_master['ds'])

    print(f"\nLoading economic indicators from: {full_indicators_csv_path}")
    df_full_indicators = pd.read_csv(full_indicators_csv_path, parse_dates=['Date'])
    df_full_indicators.rename(columns={'Date': 'ds'}, inplace=True)
    df_full_indicators['ds'] = pd.to_datetime(df_full_indicators['ds']) + pd.offsets.MonthEnd(0)
    
    # Ensure only common regressors that exist in df_full_indicators are used
    active_common_regressors = [reg for reg in common_regressor_columns if reg in df_full_indicators.columns]
    if len(active_common_regressors) != len(common_regressor_columns):
        print(f"Warning: Some specified common regressors not found. Using: {active_common_regressors}")
    df_full_indicators = df_full_indicators[['ds'] + active_common_regressors]

except FileNotFoundError as e:
    print(f"ERROR: Input data file not found: {e}")
    exit()
except ValueError as ve:
    print(f"ValueError during initial data loading: {ve}")
    exit()
except Exception as e:
    print(f"An error occurred during initial data loading: {e}")
    exit()

# Dictionary to store all forecast results
all_forecasts_df = None

# Loop through each target variable and generate a forecast
for target_column in targets_to_forecast:
    print(f"\nProcessing Forecast for: {target_column}")

    if target_column not in df_historical_master.columns:
        print(f"ERROR: Target column '{target_column}' not found in prepared data. Skipping.")
        continue

    # Prepare data for this specific target
    df_prophet_target = df_historical_master[['ds', target_column] + active_common_regressors].copy()
    df_prophet_target.rename(columns={target_column: 'y'}, inplace=True)

    historical_end_date = pd.to_datetime('2024-12-31')
    df_prophet_historical_target = df_prophet_target[df_prophet_target['ds'] <= historical_end_date].copy()

    if df_prophet_historical_target['y'].isnull().any():
        print(f"WARNING: NaNs found in target column 'y' for {target_column}. Filling with 0.")
        df_prophet_historical_target['y'].fillna(0, inplace=True)
    
    # Handle cases where the target is all zeros, which breaks multiplicative modes
    if (df_prophet_historical_target['y'] == 0).all():
        print(f"WARNING: Target column 'y' for {target_column} is all zeros. Using additive mode.")
        current_seasonality_mode = 'additive'
        current_regressor_mode = 'additive'
    else:
        current_seasonality_mode = 'multiplicative'
        current_regressor_mode = 'multiplicative'

    if len(df_prophet_historical_target) <= test_set_months:
        print(f"ERROR: Not enough historical data for {target_column} to create test set. Skipping.")
        continue

    train_df = df_prophet_historical_target.iloc[:-test_set_months]
    test_df = df_prophet_historical_target.iloc[-test_set_months:]

    print(f"Training data shape for {target_column}: {train_df.shape}")
    print(f"Test data shape for {target_column}: {test_df.shape}")

    # Model Initialization
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=current_seasonality_mode,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=1.0
    )

    for regressor in active_common_regressors:
        model.add_regressor(regressor, mode=current_regressor_mode, prior_scale=10.0)
        print(f"Added regressor for {target_column}: {regressor}, mode={current_regressor_mode}")

    try:
        print(f"Training Prophet model for {target_column}...")
        model.fit(train_df)
        print(f"Model training complete for {target_column}.")
    except Exception as e:
        print(f"Error during model training for {target_column}: {e}")
        continue

    # Evaluate Model on Test Set
    future_test_for_pred = test_df[['ds'] + active_common_regressors].copy()
    forecast_test = model.predict(future_test_for_pred)
    
    actual_y_test = test_df['y'].values
    predicted_yhat_test = forecast_test['yhat'].values

    abs_percentage_errors = np.full_like(actual_y_test, fill_value=np.nan, dtype=float)
    non_zero_mask = actual_y_test != 0
    if np.sum(non_zero_mask) > 0:
        abs_percentage_errors[non_zero_mask] = np.abs(
            (actual_y_test[non_zero_mask] - predicted_yhat_test[non_zero_mask]) / actual_y_test[non_zero_mask]
        )
    
    if np.all(np.isnan(abs_percentage_errors)): mape = np.nan
    else: mape = np.nanmean(abs_percentage_errors) * 100
    
    mae = mean_absolute_error(actual_y_test, predicted_yhat_test)
    rmse = np.sqrt(mean_squared_error(actual_y_test, predicted_yhat_test))
    
    print(f"\nModel Evaluation on Test Set for {target_column}:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Make Future Forecast
    future_dates_df = model.make_future_dataframe(periods=forecast_horizon_months, freq='ME')
    future_predict_df_target = pd.merge(future_dates_df, df_full_indicators, on='ds', how='left')
    
    if active_common_regressors and future_predict_df_target[active_common_regressors].isnull().any().any():
        print(f"Warning: NaNs found in future regressors for {target_column}. Applying ffill and bfill.")
        for reg in active_common_regressors: future_predict_df_target[reg] = future_predict_df_target[reg].ffill().bfill()
        if future_predict_df_target[active_common_regressors].isnull().any().any():
            print(f"CRITICAL WARNING: NaNs still present in future regressors for {target_column} after filling. Using 0.")
            for reg in active_common_regressors: future_predict_df_target[reg].fillna(0, inplace=True)

    print(f"\nMaking future forecast for {target_column}...")
    cols_for_prophet_predict_target = ['ds'] + active_common_regressors
    forecast_future_target = model.predict(future_predict_df_target[cols_for_prophet_predict_target])
    print(f"Future forecast complete for {target_column}.")

    # Visualize and Save Forecast Plots
    print(f"Generating forecast plots for {target_column}...")
    try:
        fig1 = model.plot(forecast_future_target)
        plt.title(f'Baseline Forecast for {target_column}')
        plt.xlabel('Date'); plt.ylabel(target_column)
        ax = fig1.gca(); ax.plot(test_df['ds'], test_df['y'], 'r.', label='Actual Test Data'); plt.legend()
        plot_filename_main = f"plot_forecast_{target_column}.png"
        plt.savefig(plot_filename_main); print(f"Saved main forecast plot to {plot_filename_main}"); plt.close(fig1)

        fig3 = model.plot_components(forecast_future_target)
        plt.suptitle(f'Forecast Components for {target_column}', y=1.02)
        plot_filename_comp = f"plot_components_{target_column}.png"
        plt.savefig(plot_filename_comp); print(f"Saved components plot to {plot_filename_comp}"); plt.close(fig3)
    except Exception as e:
        print(f"Error generating Matplotlib plots for {target_column}: {e}")

    # Store and Combine Forecast Results
    forecast_to_save = forecast_future_target[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_to_save.rename(columns={
        'yhat': f'{target_column}_forecast',
        'yhat_lower': f'{target_column}_forecast_lower',
        'yhat_upper': f'{target_column}_forecast_upper'
    }, inplace=True)

    df_actuals_target = df_prophet_historical_target[['ds', 'y']].copy()
    df_actuals_target.rename(columns={'y': f'{target_column}_actual'}, inplace=True)
    
    output_df_target = pd.merge(df_actuals_target, forecast_to_save, on='ds', how='outer')

    if all_forecasts_df is None:
        all_forecasts_df = output_df_target
    else:
        all_forecasts_df = pd.merge(all_forecasts_df, output_df_target, on='ds', how='outer')
    
    print(f"Forecast for {target_column} processed and added to combined results.")

# Save Combined Forecasts to a Single CSV
if all_forecasts_df is not None:
    combined_output_filename = "baseline_forecasts_ALL_COMPONENTS.csv"
    try:
        all_forecasts_df.sort_values(by='ds', inplace=True)
        all_forecasts_df.to_csv(combined_output_filename, index=False)
        print(f"\nSuccessfully saved ALL forecasts and actuals to: {combined_output_filename}")
        print("\nCombined forecasts final preview:")
        print(all_forecasts_df.tail(forecast_horizon_months + 3))
    except Exception as e:
        print(f"Error saving combined forecasts: {e}")
else:
    print("\nNo forecasts were generated to save in combined file.")

print(f"\nBaseline Forecast Script for ALL listed targets Completed.")