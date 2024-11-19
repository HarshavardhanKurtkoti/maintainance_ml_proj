import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_data(data):
    """
    Plot a heatmap of missing data in the dataset.
    
    Args:
        data (pd.DataFrame): The input dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()

def plot_correlation_matrix(data):
    """
    Plot a correlation matrix for the numerical columns in the dataset.
    
    Args:
        data (pd.DataFrame): The input dataset.
    """
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def plot_histograms(data, columns):
    """
    Plot histograms for selected columns in the dataset.
    
    Args:
        data (pd.DataFrame): The input dataset.
        columns (list): List of columns to plot histograms for.
    """
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Histogram of {column}')
        plt.show()

def plot_boxplot(data, columns):
    """
    Plot boxplots for selected columns to visualize the spread of the data.
    
    Args:
        data (pd.DataFrame): The input dataset.
        columns (list): List of columns to plot boxplots for.
    """
    for column in columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot of {column}')
        plt.show()

def plot_time_series(data, column, date_column):
    """
    Plot a time series of a particular column against a date column.
    
    Args:
        data (pd.DataFrame): The input dataset.
        column (str): The column to plot.
        date_column (str): The column containing date values.
    """
    data[date_column] = pd.to_datetime(data[date_column])
    plt.figure(figsize=(12, 6))
    plt.plot(data[date_column], data[column])
    plt.title(f'Time Series of {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.show()
