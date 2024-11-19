import pandas as pd

def generate_rolling_features(data, column, window_size, feature_name_prefix):
    """
    Generate rolling window features (e.g., rolling mean, rolling std).
    
    Args:
        data (pd.DataFrame): Input dataset.
        column (str): Column name for which to generate rolling features.
        window_size (int): Rolling window size.
        feature_name_prefix (str): Prefix for the generated feature names.
        
    Returns:
        pd.DataFrame: Dataset with new rolling features.
    """
    data[f'{feature_name_prefix}_rolling_mean'] = data[column].rolling(window=window_size).mean()
    data[f'{feature_name_prefix}_rolling_std'] = data[column].rolling(window=window_size).std()
    return data

def add_time_based_features(data, datetime_column):
    """
    Adds time-based features from a datetime column (e.g., hour, day, month).
    
    Args:
        data (pd.DataFrame): Input dataset.
        datetime_column (str): Name of the datetime column.
        
    Returns:
        pd.DataFrame: Dataset with additional time-based features.
    """
    data[datetime_column] = pd.to_datetime(data[datetime_column])
    data['hour'] = data[datetime_column].dt.hour
    data['day'] = data[datetime_column].dt.day
    data['month'] = data[datetime_column].dt.month
    data['year'] = data[datetime_column].dt.year
    data['day_of_week'] = data[datetime_column].dt.dayofweek
    return data

def calculate_lag_features(data, column, lag):
    """
    Calculates lag features for a specific column.
    
    Args:
        data (pd.DataFrame): Input dataset.
        column (str): Column name for which to calculate lag features.
        lag (int): Number of periods to lag.
        
    Returns:
        pd.DataFrame: Dataset with lag features.
    """
    data[f'{column}_lag_{lag}'] = data[column].shift(lag)
    return data

def create_features(data):
    """
    Example wrapper function to call feature engineering functions and generate new features.
    
    Args:
        data (pd.DataFrame): Input dataset.
        
    Returns:
        pd.DataFrame: Dataset with all new features added.
    """
    # Example: Generate rolling features
    if 'some_column' in data.columns:
        data = generate_rolling_features(data, column='some_column', window_size=3, feature_name_prefix='some_column')
    
    # Example: Add time-based features
    if 'datetime_column' in data.columns:
        data = add_time_based_features(data, datetime_column='datetime_column')
    
    # Example: Calculate lag features
    if 'another_column' in data.columns:
        data = calculate_lag_features(data, column='another_column', lag=1)
    
    return data
