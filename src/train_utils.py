import numpy as np
from sklearn.model_selection import train_test_split


class TrainUtils:
    def __init__(self, data):
        self.df = data  # Initialize the instance variable

    def process_data(self, n=10):
        # Some method to process the data
        return self.data * n

    def remove_outliers(self, column_name, method='IQR', z_threshold=3):
        """
        Remove outliers from time series data based on selected method: IQR or Z-score.
        
        Parameters:
        - df: pandas DataFrame, the time series data
        - column_name: str, the name of the column to remove outliers from
        - method: str, either 'IQR' or 'Z-score' method to detect outliers
        - z_threshold: int, the threshold for Z-score (default = 3)
        
        Returns:
        - df: pandas DataFrame, with outliers removed
        """
        if method == 'IQR':
            # Calculate IQR (Interquartile Range)
            Q1 = self.df[column_name].quantile(0.25)
            Q3 = self.df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define lower and upper bounds for detecting outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers based on IQR
            self.df = self.df[(self.df[column_name] >= lower_bound) & (self.df[column_name] <= upper_bound)]
            
        elif method == 'Z-score':
            # Calculate the Z-scores
            z_scores = (self.df[column_name] - self.df[column_name].mean()) / self.df[column_name].std()
            
            # Remove outliers based on Z-score threshold
            self.df = self.df[np.abs(z_scores) <= z_threshold]
        
        else:
            raise ValueError("Method must be either 'IQR' or 'Z-score'.")
        
        # return df_cleaned

    def data_split(self, split_param=0.2):
        return train_test_split(self.df, test_size=0.2, shuffle=False)  # No shuffle to maintain time order