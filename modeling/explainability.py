import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

def explain_model_shap(model, X_train):
    """
    Use SHAP (SHapley Additive exPlanations) to explain model predictions.

    Args:
        model: The trained model.
        X_train (pd.DataFrame): Training features.

    Returns:
        shap.Explanation: SHAP values for model explanations.
    """
    # Create a SHAP explainer object
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Plot SHAP summary
    shap.summary_plot(shap_values, X_train)
    return shap_values

def explain_model_lime(model, X_train, class_names):
    """
    Use LIME (Local Interpretable Model-agnostic Explanations) to explain predictions.

    Args:
        model: The trained model.
        X_train (pd.DataFrame): Training features.
        class_names (list): List of class names (target labels).

    Returns:
        None: Displays LIME explanations.
    """
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(), class_names=class_names, verbose=True, mode='classification')
    
    # Example of explaining a prediction for the first instance in the dataset
    explanation = explainer.explain_instance(X_train.iloc[0].values, model.predict_proba)
    explanation.show_in_notebook()

