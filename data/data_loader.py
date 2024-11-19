import pandas as pd

def load_data(file_path, **kwargs):
    """
    Loads data from a CSV file or other supported format.

    Args:
        file_path (str): Path to the data file.
        **kwargs: Additional arguments for pandas.read_csv or similar functions.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise

def load_all_data():
    """
    Loads all necessary datasets for the predictive maintenance project.
    
    Returns:
        dict: A dictionary with all loaded datasets.
    """
    file_paths = {
        'telemetry': r'files\PdM_telemetry.csv',
        'errors': r'files\PdM_errors.csv',
        'maint': r'files\PdM_maint.csv',
        'failures': r'files\PdM_failures.csv',
        'machines': r'files\PdM_machines.csv'

    }

    data = {}
    for name, path in file_paths.items():
        data[name] = load_data(path)

    return data
