import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score, f1_score, make_scorer
from src.data_loader import load_diabetes_dataset, preprocess_data, build_preprocessor
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from config import MODELS_DIR, RANDOM_STATE


def train_with_gridsearch(
    algorithm: str = "RandomForest",
    param_grid: dict = None,
    run_name: str = "GridSearch",
):
    experiment_name = "diabetes_model_comparison"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        df = load_diabetes_dataset()
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        preprocessor = build_preprocessor(feature_names)

        if algorithm == "RandomForest":
            base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
        else:
            base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

        pipeline = Pipeline(
            [("preprocessing", preprocessor), ("classifier", base_model)]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        roc_scorer = make_scorer(roc_auc_score)

        print(f"GridSearchCV pour {algorithm}...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=roc_scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time

        best_params = {
            k.replace("classifier__", ""): v
            for k, v in grid_search.best_params_.items()
        }

        y_pred = grid_search.predict(X_test)
        y_proba = grid_search.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        cv_scores = cross_val_score(
            grid_search.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc"
        )

        mlflow.log_params(
            {
                "algorithm": algorithm,
                "best_params": str(best_params),
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(feature_names),
            }
        )

        mlflow.log_metrics(
            {
                "test_roc_auc": test_metrics["roc_auc"],
                "test_recall": test_metrics["recall"],
                "test_f1_score": test_metrics["f1_score"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "cv_roc_auc_mean": cv_scores.mean(),
                "cv_roc_auc_std": cv_scores.std(),
                "training_time_seconds": round(training_time, 3),
            }
        )

        mlflow.set_tags(
            {
                "dataset": "Pima Indians Diabetes",
                "task": "binary_classification",
                "framework": "scikit-learn",
                "search_type": "GridSearchCV",
            }
        )

        input_example = pd.DataFrame(X_test[:3], columns=feature_names)
        output_example = pd.DataFrame(y_pred[:3], columns=["Outcome"])
        signature = infer_signature(input_example, output_example)

        mlflow.sklearn.log_model(
            grid_search.best_estimator_,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=f"{algorithm.lower()}_best_model",
        )

        cm_path = plot_confusion_matrix(y_test, y_pred, run_name, MODELS_DIR)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        roc_path = plot_roc_curve(y_test, y_proba, run_name, MODELS_DIR)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        print(f"\n{run_name} - Best params: {best_params}")
        print(
            f"{run_name} - CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )
        print(
            f"{run_name} - Test ROC-AUC: {test_metrics['roc_auc']:.4f}, Recall: {test_metrics['recall']:.4f}"
        )

        return grid_search.best_estimator_, test_metrics, run.info.run_id


if __name__ == "__main__":
    rf_param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": ["balanced", None],
    }

    print("=" * 60)
    print("DEFI: GridSearch + GradientBoosting")
    print("=" * 60)

    train_with_gridsearch(
        algorithm="RandomForest", param_grid=rf_param_grid, run_name="RF_GridSearch"
    )

    gb_param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__min_samples_split": [2, 5],
    }

    train_with_gridsearch(
        algorithm="GradientBoosting", param_grid=gb_param_grid, run_name="GB_GridSearch"
    )
