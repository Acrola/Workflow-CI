import mlflow
import pandas as pd
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split

def train_and_log_model(X_train, y_train, X_test, y_test, params, model_name="RandomForestClassifier"):
    """
    Trains a RandomForestClassifier with given parameters and manually logs results to MLflow.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        params (dict): Dictionary of hyperparameters for the model.
        model_name (str): A name for the model, for logging purposes.

    Returns:
        dict: A dictionary containing key metrics and the trained model.
    """
    print(f"  Training model with params: {params}")

    # --- 1. Manually Log Hyperparameters (same as autolog) ---
    mlflow.log_param("bootstrap", params.get("bootstrap", True))
    mlflow.log_param("ccp_alpha", params.get("ccp_alpha", 0.0))
    mlflow.log_param("class_weight", params.get("class_weight", None))
    mlflow.log_param("criterion", params.get("criterion", "gini"))
    mlflow.log_param("max_depth", params.get("max_depth", None))
    mlflow.log_param("max_features", params.get("max_features", "sqrt"))
    mlflow.log_param("max_leaf_nodes", params.get("max_leaf_nodes", None))
    mlflow.log_param("min_samples_leaf", params.get("min_samples_leaf", 1))
    mlflow.log_param("min_samples_split", params.get("min_samples_split", 2))
    mlflow.log_param("min_impurity_decrease", params.get("min_impurity_decrease", 0.0))
    mlflow.log_param("min_weight_fraction_leaf", params.get("min_weight_fraction_leaf", 0.0))
    mlflow.log_param("monotonic_cst", params.get("monotonic_cst", None))
    mlflow.log_param("n_estimators", params.get("n_estimators", 100))
    mlflow.log_param("n_jobs", params.get("n_jobs", None))
    mlflow.log_param("oob_score", params.get("oob_score", False))
    mlflow.log_param("verbose", params.get("verbose", 0))
    mlflow.log_param("warm_start", params.get("warm_start", False))

    # Initialize the RandomForestClassifier with the provided parameters
    model = RandomForestClassifier(**params)

    # --- 2. Manually Log Training Time (1st Additional Metric) ---
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_duration = end_time - start_time
    mlflow.log_metric("training_duration_seconds", training_duration)
    print(f"  Training took {training_duration:.2f} seconds.")

    # Predict probabilities for log_loss and roc_auc
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # --- 3. Manually Log Metrics (same as autolog) ---
    mlflow.log_metric("test_accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("test_log_loss", log_loss(y_test, y_pred_proba))
    mlflow.log_metric("test_precision_score", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall_score", recall_score(y_test, y_pred))
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba[:, 1]))
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred):.4f}, ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
    # Also log training metrics
    mlflow.log_metric("training_accuracy_score", accuracy_score(y_train, model.predict(X_train)))
    mlflow.log_metric("training_f1_score", f1_score(y_train, model.predict(X_train)))
    mlflow.log_metric("training_log_loss", log_loss(y_train, model.predict_proba(X_train)))
    mlflow.log_metric("training_precision_score", precision_score(y_train, model.predict(X_train)))
    mlflow.log_metric("training_recall_score", recall_score(y_train, model.predict(X_train)))
    mlflow.log_metric("training_roc_auc", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    print(f"  Training Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}, F1: {f1_score(y_train, model.predict(X_train)):.4f}, ROC AUC: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.4f}")


    # --- 4. Manually Log Specificity (2nd Additional Metric) ---
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # Handle potential division by zero if (tn + fp) is 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    mlflow.log_metric("test_specificity", specificity)
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred):.4f}, Specificity: {specificity:.4f}")


    # --- 5. Log the Model Artifact ---
    # Save model under the current MLflow run as an artifact.
    mlflow.sklearn.log_model(model, "model", signature=mlflow.models.infer_signature(X_test, y_pred))
    print("  Model artifact logged.")

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

    with mlflow.start_run(run_name="CI_Automated_Retrain_Run_Dynamic_Params") as ci_run:
        print("--- Starting CI Automated Retraining with Dynamic Parameters ---")

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
            runs = client.search_runs(
                experiment_names=[TUNING_EXPERIMENT_NAME],
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
                        k: (int(v) if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split'] else v)
                        for k, v in best_child_run.data.params.items()
                        if k in model_params_to_extract
                    }
                    print(f"Retrieved best parameters from run {best_child_run_id}: {best_retrain_params}")
                    mlflow.log_param("retrained_from_best_run_id", best_child_run_id) # Log source run ID

        except Exception as e:
            print(f"An error occurred while retrieving best parameters from MLflow: {e}")
            print("Proceeding with default parameters as a fallback.")
            best_retrain_params = {
                'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1,
                'max_features': 'sqrt', 'min_samples_split': 2
            }

        # Log retrieved parameters for this CI run
        mlflow.log_params(best_retrain_params)

        # 3. Call training function with the retrieved parameters
        train_and_log_model(X_train, y_train, X_test, y_test, best_retrain_params)
        print("--- CI Automated Retraining Complete ---")
        print(ci_run.info.run_id)  # Print the current run's ID


