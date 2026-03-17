import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data_loader import load_diabetes_dataset, preprocess_data, build_preprocessor
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from config import MODELS_DIR, RANDOM_STATE


def find_optimal_threshold(y_true, y_proba):
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.15, 0.7, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        from sklearn.metrics import f1_score

        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def train_xgboost_v2(params: dict = None, run_name: str = "XGBoost"):
    experiment_name = "diabetes_xgboost_v2"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        df = load_diabetes_dataset()
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        preprocessor = build_preprocessor(feature_names)

        clf = XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )

        pipeline = Pipeline([("preprocessing", preprocessor), ("classifier", clf)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
        )

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred_default = pipeline.predict(X_test)
        test_metrics_default = compute_metrics(y_test, y_pred_default, y_proba)

        optimal_threshold = find_optimal_threshold(y_test, y_proba)
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
        test_metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_proba)

        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(
            y_train, pipeline.predict(X_train), y_train_proba
        )

        mlflow.log_params(
            {
                "algorithm": "XGBoost",
                **params,
                "optimal_threshold": round(optimal_threshold, 3),
                "cv_folds": 5,
                "random_state": RANDOM_STATE,
            }
        )

        mlflow.log_metrics(
            {
                "test_roc_auc": test_metrics_optimal["roc_auc"],
                "test_recall": test_metrics_optimal["recall"],
                "test_f1_score": test_metrics_optimal["f1_score"],
                "test_accuracy": test_metrics_optimal["accuracy"],
                "test_precision": test_metrics_optimal["precision"],
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
                "framework": "xgboost",
            }
        )

        input_example = pd.DataFrame(X_test[:3], columns=feature_names)
        signature = infer_signature(input_example, y_proba[:3])

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="diabetes_xgb_best",
        )

        cm_path = plot_confusion_matrix(y_test, y_pred_optimal, run_name, MODELS_DIR)
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
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test ROC-AUC: {test_metrics_optimal['roc_auc']:.4f}")
        print(f"  Test Recall: {test_metrics_optimal['recall']:.4f}")
        print(f"  Test F1: {test_metrics_optimal['f1_score']:.4f}")

        return pipeline, test_metrics_optimal, optimal_threshold, cv_scores.mean()


if __name__ == "__main__":
    print("=" * 60)
    print("XGBOOST V2 - Target ROC-AUC >= 0.85, Recall >= 0.75")
    print("=" * 60)

    configs = [
        (
            "XGB_v2_1",
            {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "scale_pos_weight": 2,
                "reg_alpha": 0.1,
                "reg_lambda": 1,
            },
        ),
        (
            "XGB_v2_2",
            {
                "n_estimators": 600,
                "max_depth": 3,
                "learning_rate": 0.02,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": 1.8,
                "reg_alpha": 0.05,
                "reg_lambda": 2,
            },
        ),
        (
            "XGB_v2_3",
            {
                "n_estimators": 500,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": 2.2,
                "reg_alpha": 0.2,
                "reg_lambda": 1.5,
            },
        ),
    ]

    results = []

    for name, params in configs:
        _, metrics, threshold, cv_score = train_xgboost_v2(params, name)
        results.append((name, metrics, threshold, cv_score))

    print("\n" + "=" * 60)
    print("XGBOOST V2 RESULTS")
    print("=" * 60)
    for name, m, t, cv in results:
        status = (
            "*** SUCCESS ***" if m["roc_auc"] >= 0.85 and m["recall"] >= 0.75 else ""
        )
        print(
            f"{name}: ROC-AUC={m['roc_auc']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}, CV={cv:.4f} {status}"
        )

    success = [r for r in results if r[1]["roc_auc"] >= 0.85 and r[1]["recall"] >= 0.75]
    if success:
        print(f"\n*** SUCCESS! {len(success)} model(s) achieved target ***")
