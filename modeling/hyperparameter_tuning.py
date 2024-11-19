# modeling/hyperparameter_tuning.py
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def tune_model():
    # Load data
    data = load_iris()
    X = data.data
    y = data.target

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15]
    }

    # Set up MLflow experiment tracking
    with mlflow.start_run():

        # Initialize model
        model = RandomForestClassifier()

        # Grid search for best parameters
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X, y)

        # Log best parameters and score
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_metric('best_score', grid_search.best_score_)

        print(f'Best Score: {grid_search.best_score_}')
        print(f'Best Params: {grid_search.best_params_}')

if __name__ == '__main__':
    tune_model()
