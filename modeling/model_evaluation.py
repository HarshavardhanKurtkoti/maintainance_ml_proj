# modeling/model_evaluation.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data.preprocessing import load_data

def evaluate_model():
    # Load the data
    X, y = load_data()

    # Load the model from MLflow (use the model's artifact URI)
    model_uri = "runs:/<RUN_ID>/random_forest_model"  # Replace <RUN_ID> with the actual run ID from MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # Split data and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

if __name__ == '__main__':
    evaluate_model()
