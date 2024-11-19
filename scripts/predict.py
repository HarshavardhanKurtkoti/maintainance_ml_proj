# scripts/predict.py
import mlflow
import mlflow.sklearn
from data.preprocessing import load_data

def predict():
    # Load model from MLflow
    model_uri = "runs:/<RUN_ID>/random_forest_model"  # Replace <RUN_ID> with the actual run ID from MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # Load test data
    X, _ = load_data()  # Assuming you only need features for prediction

    # Make predictions
    predictions = model.predict(X)
    print(predictions)

if __name__ == '__main__':
    predict()
