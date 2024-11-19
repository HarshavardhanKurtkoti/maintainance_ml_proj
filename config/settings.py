import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Directory for data
MODELS_DIR = os.path.join(BASE_DIR, 'models')  # Directory for models
EXPLORATION_DIR = os.path.join(BASE_DIR, 'exploration')  # Directory for exploration

# Hyperparameters and constants
SEED = 42
TEST_SIZE = 0.2
RANDOM_STATE = SEED

# Model configurations
MODEL_TYPE = 'random_forest'  # Specify model type
HYPERPARAMETERS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

# Logging settings
LOG_LEVEL = 'INFO'

# MLFlow settings
MLFLOW_TRACKING_URI = "http://localhost:5000"  # URI for MLFlow tracking server
MLFLOW_EXPERIMENT_NAME = "predictive-maintenance-experiment"
