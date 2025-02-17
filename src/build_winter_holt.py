import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


class TrainWinter:

    def __init__(self):
        pass
        
    def build_model(self, train_data):
        """
        Build Forecasting model using Prophet
        """
        # Holt-Winters model (Additive) for seasonal data
        model = ExponentialSmoothing(
            train_data['y'], 
            trend='add',   # Use additive trend (you can also use 'mul' for multiplicative)
            seasonal='add',  # Use additive seasonality (use 'mul' for multiplicative seasonality)
            seasonal_periods=12*24  # Periods in a seasonal cycle, e.g., monthly seasonality if it's annual data (12 months)
        )

        model.fit(train_data)
        return model

    def predict(self, model, periods):
        # Make future predictions (including the test period)
        # Forecast for the next 12 periods (e.g., next 12 months or 12 time steps)
        forecast = model.forecast(steps=periods)
        return forecast

    def evaluate_model(self, forecast, test_data):
        # Merge forecasted values with actual test data for evaluation
        forecast_test = forecast.tail(len(test_data))  # Get forecast for the test period
        comparison = test_data[['ds', 'y']].merge(forecast_test[['ds', 'yhat']], on='ds')
        
        # Evaluate accuracy using scikit-learn error metrics
        mae = mean_absolute_error(comparison['y'], comparison['yhat'])
        mse = mean_squared_error(comparison['y'], comparison['yhat'])
        rmse = mean_squared_error(comparison['y'], comparison['yhat'], squared=False)
        mape = mean_absolute_percentage_error(comparison['y'], comparison['yhat']) * 100
        
        # Print the results
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

        # Plot the actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(comparison['ds'], comparison['y'], label='Actual', color='blue')
        plt.plot(comparison['ds'], comparison['yhat'], label='Forecast', color='red', linestyle='--')
        plt.title('Actual vs Forecasted Values')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        return mae, mse, rmse, mape
        