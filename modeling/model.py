# modeling/model.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming your data is loaded and preprocessed
from data.preprocessing import load_data  # Load your data function

# Set up MLflow tracking
mlflow.set_tracking_uri('http://localhost:5000')  # You can change the URI as needed
mlflow.set_experiment('predictive-maintenance-experiment')

# Function to train model and track experiment
def train_model():
    # Load data
    X, y = load_data()  # Make sure to define this function in your data folder

    # Split data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start an MLflow run
    with mlflow.start_run():

        # Model initialization
        model = RandomForestClassifier(n_estimators=100, max_depth=10)

        # Log hyperparameters
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('max_depth', 10)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric('accuracy', accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, 'random_forest_model')

        print(f'Model Accuracy: {accuracy}')

if __name__ == '__main__':
    train_model()
