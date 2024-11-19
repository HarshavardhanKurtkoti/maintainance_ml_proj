# data/split.py
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The input dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Test labels.
    """
    X = data.drop(columns=['target'])  # Assuming the target column is named 'target'
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# You could also add time-based splitting functionality here, depending on your needs
