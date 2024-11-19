import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """
    Loads the dataset from a given file path.
    
    Args:
        file_path (str): Path to the dataset file (e.g., CSV or Excel).
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)  # Adjust for your file type (CSV, Excel, etc.)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(data, strategy='mean', fill_value=None):
    if strategy == 'mean':
        return data.fillna(data.mean())
    elif strategy == 'median':
        return data.fillna(data.median())
    elif strategy == 'constant':
        fill_value = fill_value if fill_value is not None else 'Unknown'  # Default value
        return data.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy or fill_value not provided for 'constant'.")

def scale_features(data, method='standard', columns=None):
    if columns is None:
        columns = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    scaled_data = scaler.fit_transform(data[columns])
    data[columns] = scaled_data
    return data

def encode_categorical(data, columns):
    for column in columns:
        if data[column].dtype == 'object' or data[column].dtype.name == 'category':
            data = pd.get_dummies(data, columns=[column])
        else:
            print(f"Column {column} is not categorical, skipping encoding.")
    return data

def preprocess_data(data, categorical_columns=None, strategy='mean', scaling_method='standard', fill_value=None):
    # Handle missing values
    data = handle_missing_values(data, strategy=strategy, fill_value=fill_value)
    
    # Encode categorical features
    if categorical_columns:
        data = encode_categorical(data, columns=categorical_columns)
    
    # Scale numerical features
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data = scale_features(data, method=scaling_method, columns=numerical_columns)
    
    return data
