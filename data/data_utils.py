import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def time_to_datetime(df, time_column):
    """
    Convert a time column to datetime format in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        time_column (str): The name of the column to convert.
    
    Returns:
        pd.DataFrame: The DataFrame with the time column converted to datetime.
    """
    df[time_column] = pd.to_datetime(df[time_column])
    return df

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (pd.DataFrame): The input dataset.
        strategy (str): Strategy to handle missing values. Can be 'mean', 'median', or 'drop'.
        
    Returns:
        pd.DataFrame: The dataset with missing values handled.
    """
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'drop':
        return data.dropna()  # You could add axis argument here if needed: data.dropna(axis=0 or 1)
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', or 'drop'.")

def convert_categorical_to_numerical(data, columns):
    """
    Convert categorical columns in the dataset to numerical representations.
    
    Args:
        data (pd.DataFrame): The input dataset.
        columns (list): List of column names to be converted.
        
    Returns:
        pd.DataFrame: The dataset with categorical columns converted to numerical values.
    """
    for column in columns:
        # Handling NaN values in categorical columns before conversion
        data[column] = data[column].fillna('Unknown').astype('category').cat.codes
    return data

def scale_features(data, scaler=None):
    """
    Scale the features in the dataset using Min-Max Scaling or Standard Scaling.
    
    Args:
        data (pd.DataFrame): The input dataset.
        scaler (sklearn scaler, optional): The scaler to use. If None, uses MinMaxScaler.
        
    Returns:
        pd.DataFrame: The scaled dataset.
    """
    if scaler is None:
        scaler = MinMaxScaler()
    
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)
