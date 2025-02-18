import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TrainXGBoost:

    def __init__(self):
        pass
        
    def build_xgb_model(self, train_data):
        """
        Build Forecasting model using XGBoost
        """
        features = ['dow', 'doy', 'year', 'month', 'quarter', 'hour', 
            'weekday', 'woy', 'dom', 'season', 'lag_1', 'lag_2', 'lag_3']
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        X_train = train_data[features]
        y_train = train_data['y']
        model.fit(X_train, y_train)
        return model

    def predict(self, model, periods, X_test):
        # Make future predictions (including the test period)
        y_pred = model.predict(X_test)
        return y_pred

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
        