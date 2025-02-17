import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TrainProphet:

    def __init__(self):
        pass
        
    def build_prophet_model(self, train_data):
        """
        Build Forecasting model using Prophet
        """
        # Initialize and fit the Prophet model
        model = Prophet(seasonality_mode='multiplicative', changepoints=['2014-01-01'])
        
        # Add custom seasonality for day of the week
        model.add_seasonality(name='weekly', period=7, fourier_order=4)
        
        # Add custom seasonality for month of the year
        model.add_seasonality(name='monthly', period=365.25/12, fourier_order=6)
        
        # Add custom seasonality for month of the year
        model.add_seasonality(name='daily', period=1, fourier_order=10)

        # Add custom seasonality for month of the year
        model.add_seasonality(name='hourly', period=24, fourier_order=8)

        model.fit(train_data)
        return model

    def predict(self, model, periods):
        # Make future predictions (including the test period)
        future = model.make_future_dataframe(periods=10, freq='H')
        forecast = model.predict(future)
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
        