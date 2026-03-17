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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
from src.data_loader import load_diabetes_dataset, preprocess_data, build_preprocessor
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from config import MODELS_DIR, RANDOM_STATE


def train_optimized_model(
    algorithm: str = "GradientBoosting",
    params: dict = None,
    run_name: str = "Optimized",
):
    experiment_name = "diabetes_optimized"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        df = load_diabetes_dataset()
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        preprocessor = build_preprocessor(feature_names)

        if algorithm == "RandomForest":
            clf = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            clf = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)

        pipeline = Pipeline([("preprocessing", preprocessor), ("classifier", clf)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
        )

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        y_train_pred = pipeline.predict(X_train)
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)

        mlflow.log_params(
            {
                "algorithm": algorithm,
                **params,
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                "test_roc_auc": test_metrics["roc_auc"],
                "test_recall": test_metrics["recall"],
                "test_f1_score": test_metrics["f1_score"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "train_roc_auc": train_metrics["roc_auc"],
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
            }
        )

        input_example = pd.DataFrame(X_test[:3], columns=feature_names)
        signature = infer_signature(input_example, y_proba[:3])

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="diabetes_optimized_model",
        )

        cm_path = plot_confusion_matrix(y_test, y_pred, run_name, MODELS_DIR)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        roc_path = plot_roc_curve(y_test, y_proba, run_name, MODELS_DIR)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        feature_importance = pipeline.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=False)
        importance_path = os.path.join(MODELS_DIR, f"feature_importance_{run_name}.csv")
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="reports")

        print(f"\n{run_name}:")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(
            f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}, Recall: {test_metrics['recall']:.4f}"
        )

        return pipeline, test_metrics


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZATION CHALLENGE")
    print("=" * 60)

    configs = [
        (
            "GB_opt1",
            "GradientBoosting",
            {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 5,
            },
        ),
        (
            "GB_opt2",
            "GradientBoosting",
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_samples_split": 3,
            },
        ),
        (
            "RF_opt",
            "RandomForest",
            {
                "n_estimators": 300,
                "max_depth": 10,
                "class_weight": "balanced",
                "min_samples_split": 3,
            },
        ),
    ]

    best_model = None
    best_score = 0
    best_name = ""

    for name, algo, params in configs:
        _, metrics = train_optimized_model(algo, params, name)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = metrics
            best_name = name

    print(f"\nBest model: {best_name} with ROC-AUC = {best_score:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
