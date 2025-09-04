import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Configuration
input_data_path = "prepared_cash_flow_data.csv"
full_indicators_csv_path = "mock_economic_indicators.csv"

target_column = 'inflow_operating_revenue'

# Define which economic indicators to use as regressors
regressor_columns = [
    'Economic_Activity_Index',
    'Consumer_Confidence_Index',
    'Interest_Rate_Proxy',
    'EV_Component_Cost_Index'
]

forecast_horizon_months = 24
test_set_months = 6

print(f"Starting Baseline Forecast for: {target_column}")

# 1. Load and Prepare Data
try:
    # Load historical data with target variable and regressors
    df_historical_data = pd.read_csv(input_data_path)
    
    # Identify and rename date column to 'ds' for Prophet
    if 'statement_date' in df_historical_data.columns:
        df_historical_data.rename(columns={'statement_date': 'ds'}, inplace=True)
    elif 'balance_date' in df_historical_data.columns:
        df_historical_data.rename(columns={'balance_date': 'ds'}, inplace=True)
    elif df_historical_data.columns[0].lower() == 'date' or df_historical_data.columns[0].lower().endswith('_date'):
         df_historical_data.rename(columns={df_historical_data.columns[0]: 'ds'}, inplace=True)
    elif 'Unnamed: 0' in df_historical_data.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_historical_data['Unnamed: 0'], errors='coerce')):
            df_historical_data.rename(columns={'Unnamed: 0': 'ds'}, inplace=True)
    else:
        raise ValueError("Date column ('ds') not found in prepared_cash_flow_data.csv.")
    
    df_historical_data['ds'] = pd.to_datetime(df_historical_data['ds'])
    df_historical_data.rename(columns={target_column: 'y'}, inplace=True)

    # Prepare historical data (up to Dec 2024) for training and testing
    historical_end_date = pd.to_datetime('2024-12-31')
    df_prophet_train_test = df_historical_data[df_historical_data['ds'] <= historical_end_date].copy()
    
    # Filter to only necessary columns and active regressors
    prophet_train_test_cols = ['ds', 'y']
    active_regressors = []
    for reg in regressor_columns:
        if reg in df_prophet_train_test.columns:
            prophet_train_test_cols.append(reg)
            active_regressors.append(reg)
        else:
            print(f"Warning: Regressor '{reg}' not found in historical data and will be ignored.")
    regressor_columns = active_regressors

    df_prophet_train_test = df_prophet_train_test[prophet_train_test_cols]

    if df_prophet_train_test['y'].isnull().any():
        print(f"WARNING: NaNs found in target column 'y'. Filling with 0.")
        df_prophet_train_test['y'].fillna(0, inplace=True)

    # Load the full set of economic indicators for future predictions
    print(f"\nLoading FULL economic indicators from: {full_indicators_csv_path}")
    df_full_indicators = pd.read_csv(full_indicators_csv_path, parse_dates=['Date'])
    df_full_indicators.rename(columns={'Date': 'ds'}, inplace=True)
    df_full_indicators['ds'] = pd.to_datetime(df_full_indicators['ds']) + pd.offsets.MonthEnd(0)
    df_full_indicators = df_full_indicators[['ds'] + regressor_columns]

    print("Data loaded and prepared.")

except FileNotFoundError as e:
    print(f"ERROR: Input data file not found: {e}")
    exit()
except ValueError as ve:
    print(f"ValueError during data preparation: {ve}")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# 2. Split into Training and Test Sets
if len(df_prophet_train_test) <= test_set_months:
    print("ERROR: Not enough historical data to create a test set.")
    exit()

train_df = df_prophet_train_test.iloc[:-test_set_months]
test_df = df_prophet_train_test.iloc[-test_set_months:]

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# 3. Train Prophet Model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=1.0
)

for regressor in regressor_columns:
    model.add_regressor(regressor, mode='multiplicative', prior_scale=10.0)
    print(f"Added regressor: {regressor}")

try:
    print("\nTraining Prophet model...")
    model.fit(train_df)
    print("Model training complete.")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# 4. Evaluate Model on Test Set
future_test_for_pred = test_df[['ds'] + regressor_columns].copy()
forecast_test = model.predict(future_test_for_pred)

# Calculate evaluation metrics (MAE, RMSE, MAPE)
actual_y_test = test_df['y'].values
predicted_yhat_test = forecast_test['yhat'].values

abs_percentage_errors = np.full_like(actual_y_test, fill_value=np.nan, dtype=float)
non_zero_mask = actual_y_test != 0
if np.sum(non_zero_mask) > 0:
    abs_percentage_errors[non_zero_mask] = np.abs(
        (actual_y_test[non_zero_mask] - predicted_yhat_test[non_zero_mask]) / actual_y_test[non_zero_mask]
    )

if np.all(np.isnan(abs_percentage_errors)):
    mape = np.nan
else:
    mape = np.nanmean(abs_percentage_errors) * 100

mae = mean_absolute_error(actual_y_test, predicted_yhat_test)
rmse = np.sqrt(mean_squared_error(actual_y_test, predicted_yhat_test))

print("\nModel Evaluation on Test Set:")
print(f"Target Variable: {target_column}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# 5. Make Future Forecast
# Create a dataframe for future dates and merge with future regressor values
future_dates_df = model.make_future_dataframe(periods=forecast_horizon_months, freq='ME')
future_predict_df = pd.merge(future_dates_df, df_full_indicators, on='ds', how='left')

# Check for and handle any potential missing future regressor values
if regressor_columns and future_predict_df[regressor_columns].isnull().any().any():
    print("\nWARNING: NaNs found in future regressor columns after merge. Filling with 0 as a fallback.")
    print("Ensure mock_economic_indicators.csv covers the full forecast horizon.")
    for reg in regressor_columns:
        future_predict_df[reg].fillna(0, inplace=True)

print(f"\nMaking future forecast for {forecast_horizon_months} months...")
cols_for_prophet_predict = ['ds'] + regressor_columns
forecast_future = model.predict(future_predict_df[cols_for_prophet_predict])
print("Future forecast complete.")

# 6. Visualize Forecast
print("\nGenerating forecast plots...")
fig1 = model.plot(forecast_future)
plt.title(f'Baseline Forecast for {target_column}')
plt.xlabel('Date')
plt.ylabel(target_column)
ax = fig1.gca()
ax.plot(test_df['ds'], test_df['y'], 'r.', label='Actual Test Data')
plt.legend()
plt.show()

try:
    fig2 = plot_plotly(model, forecast_future)
    fig2.update_layout(title=f'Interactive Baseline Forecast for {target_column}')
    fig2.add_scatter(x=test_df['ds'], y=test_df['y'], mode='markers', name='Actual Test Data', marker=dict(color='red'))
    fig2.show()
except Exception as e:
    print(f"Could not generate interactive plotly plot: {e}")

try:
    fig3 = model.plot_components(forecast_future)
    plt.suptitle(f'Forecast Components for {target_column}', y=1.02)
    plt.show()
except Exception as e:
    print(f"Could not generate components plot: {e}")

try:
    if regressor_columns:
        fig4 = plot_components_plotly(model, forecast_future)
        fig4.update_layout(title=f'Interactive Forecast Components for {target_column}')
        fig4.show()
except Exception as e:
    print(f"Could not generate interactive components plot: {e}")

# 7. Store Forecast
forecast_to_save = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_to_save.rename(columns={
    'yhat': f'{target_column}_forecast',
    'yhat_lower': f'{target_column}_forecast_lower',
    'yhat_upper': f'{target_column}_forecast_upper'
}, inplace=True)

df_actuals_to_merge = df_prophet_train_test[['ds', 'y']].copy()
df_actuals_to_merge.rename(columns={'y': f'{target_column}_actual'}, inplace=True)

output_df = pd.merge(df_actuals_to_merge, forecast_to_save, on='ds', how='outer')
output_filename = f"baseline_forecast_{target_column}.csv"

try:
    output_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved forecast and actuals to: {output_filename}")
except Exception as e:
    print(f"Error saving forecast: {e}")

print(f"\nBaseline Forecast Script for {target_column} Completed.")