import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))  # Add the data directory to path
import argparse
from data.data_loader import load_data
from data.preprocessing import preprocess_data
from modeling.model import train_model
from modeling.model_evaluation import evaluate_model
import config.settings as settings  # Import the settings module

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a predictive maintenance model.')
    parser.add_argument('--config', type=str, help='Path to config file', default='config/settings.py')
    args = parser.parse_args()

    # Load configuration settings
    print(f"Using configuration: {args.config}")

    # Load and preprocess data
    data = load_data(settings.DATA_DIR)  # Use settings.DATA_DIR directly
    preprocessed_data = preprocess_data(data)

    # Train the model
    model = train_model(preprocessed_data)

    # Evaluate the model
    evaluate_model(model, preprocessed_data)

if __name__ == "__main__":
    main()
