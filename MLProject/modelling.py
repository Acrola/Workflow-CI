import mlflow
import mlflow.sklearn
import pandas as pd
import time
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split

def train_and_log_model(X_train, y_train, X_test, y_test, params, model_name="RandomForestClassifier"):
    """
    Trains a RandomForestClassifier with given parameters and logs results to MLflow.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        params (dict): Dictionary of hyperparameters for the model.
        model_name (str): A name for the model, for logging purposes.

    Returns:
        dict: A dictionary containing key metrics and the trained model.
    """
    print(f"  Training model with params: {params}", file=sys.stderr)

    # --- 1. Log Hyperparameters and metrics---
    mlflow.autolog()

    # Initialize the RandomForestClassifier with the provided parameters
    model = RandomForestClassifier(random_state=42, **params)

    # --- 2. Manually Log Training Time (1st Additional Metric) ---
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_duration = end_time - start_time
    mlflow.log_metric("training_duration_seconds", training_duration)
    print(f"  Training took {training_duration:.2f} seconds.", file=sys.stderr)

    # --- 3. Predict probabilities for log_loss and roc_auc ---
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # --- 4. Manually Log Specificity (2nd Additional Metric) ---
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # Handle potential division by zero if (tn + fp) is 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    mlflow.log_metric("test_specificity", specificity)
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred):.4f}, Specificity: {specificity:.4f}", file=sys.stderr)

    # --- 5. Save the model locally for later use ---
    mlflow.sklearn.save_model(model, "../model")
    print("  Model saved locally as MLflow model in ../model/", file=sys.stderr)

    # Return key results to the tuning script for comparison
    return {
        "model": model,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba[:, 1])
    }

# --- Automated Retraining ---
if __name__ == "__main__":
    # Define your MLflow Experiment Name for CI retraining runs
    CI_EXPERIMENT_NAME = "Diabetes_Prediction_CI_Retraining"
    mlflow.set_experiment(CI_EXPERIMENT_NAME)

    # Define experiment name and parent run name of tuning results
    TUNING_EXPERIMENT_NAME = "Diabetes_Prediction_Hyperparameter_Tuning"
    PARENT_TUNING_RUN_NAME = "ParameterGrid_Hyperparameter_Tuning_Parent_Run"

    # Configure MLflow client (using environment variables from CI/local setup)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    client = mlflow.tracking.MlflowClient()

    print("--- Starting CI Automated Retraining with Dynamic Parameters ---", file=sys.stderr)

    # 1. Load the dataset
    try:
        data = pd.read_csv("diabetes_preprocessing.csv")
    except FileNotFoundError:
        print("Dataset not found. Please ensure 'diabetes_preprocessing.csv' is in the MLProject folder.")
        exit(1)

    X = data.drop('Diabetes_binary', axis=1)
    y = data['Diabetes_binary']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Retrieve best parameters from the MLflow Tracking Server (DagsHub)
    best_retrain_params = {}
    try:
        # Search for the latest parent tuning run
        experiment = client.get_experiment_by_name(TUNING_EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{PARENT_TUNING_RUN_NAME}' AND attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if not runs:
            print(f"Warning: No parent tuning run found with name '{PARENT_TUNING_RUN_NAME}' in experiment '{TUNING_EXPERIMENT_NAME}'.")
            print("Proceeding with default parameters as a fallback.")
            # Fallback to a reasonable set of default parameters if no tuning run is found
            best_retrain_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'min_samples_split': 2,
            }
        else:
            parent_run = runs[0]
            best_child_run_id = parent_run.data.params.get("best_model_run_id")

            if not best_child_run_id:
                print(f"Warning: Parent run {parent_run.info.run_id} does not contain 'best_model_run_id' parameter.")
                print("Proceeding with default parameters as a fallback.")
                best_retrain_params = {
                    'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1,
                    'max_features': 'sqrt', 'min_samples_split': 2
                }
            else:
                # Fetch the best child run to get its parameters
                best_child_run = client.get_run(best_child_run_id)
                # Filter for actual model hyperparameters, and convert types if necessary
                model_params_to_extract = [
                    'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features',
                    'min_samples_split', 'bootstrap', 'ccp_alpha', 'class_weight',
                    'criterion', 'max_leaf_nodes', 'min_impurity_decrease',
                    'min_weight_fraction_leaf', 'monotonic_cst', 'n_jobs',
                    'oob_score', 'verbose', 'warm_start'
                ]
                best_retrain_params = {
                    k: (
                        int(v) if k in [
                            'n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split',
                            'max_leaf_nodes', 'n_jobs', 'verbose'
                        ] and v not in [None, 'None', '']
                        else float(v) if k in [
                            'ccp_alpha', 'min_impurity_decrease', 'min_weight_fraction_leaf'
                        ] and v not in [None, 'None', '']
                        else True if v == 'True'
                        else False if v == 'False'
                        else None if v in [None, 'None', '']
                        else v
                    )
                    for k, v in best_child_run.data.params.items()
                    if k in model_params_to_extract
                }
                print(f"Retrieved best parameters from run {best_child_run_id}: {best_retrain_params}", file=sys.stderr)
                mlflow.log_param("retrained_from_best_run_id", best_child_run_id) # Log source run ID

    except Exception as e:
        print(f"An error occurred while retrieving best parameters from MLflow: {e}", file=sys.stderr)
        print("Proceeding with default parameters as a fallback.", file=sys.stderr)
        best_retrain_params = {
            'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1,
            'max_features': 'sqrt', 'min_samples_split': 2
        }

    # Remove 'random_state' from best_retrain_params if present
    best_retrain_params.pop('random_state', None)

    # Log retrieved parameters for this CI run
    mlflow.log_params(best_retrain_params)

    # 3. Call training function with the retrieved parameters
    train_and_log_model(X_train, y_train, X_test, y_test, best_retrain_params)
    print("--- CI Automated Retraining Complete ---", file=sys.stderr)

    # Print the current run id (for workflow parsing)
    active_run = mlflow.active_run()
    if active_run:
        print(active_run.info.run_id)
