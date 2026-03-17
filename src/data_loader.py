import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, RAW_DIR


def load_diabetes_dataset() -> pd.DataFrame:
    """
    Charge le dataset Pima Indians Diabetes depuis OpenML.
    Retourne un DataFrame avec la colonne cible 'Outcome'.
    """
    print("Chargement du dataset Pima Indians Diabetes...")
    dataset = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = dataset.frame.copy()
    df["Outcome"] = (df["class"] == "tested_positive").astype(int)
    df = df.drop(columns=["class"])
    print(f"Dataset charge : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"Distribution : {df['Outcome'].value_counts().to_dict()}")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Prepare les donnees : remplace les zeros aberrants,
    realise le split train/test stratifie.
    Retourne : X_train, X_test, y_train, y_test, feature_names
    """
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    feature_names = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_names].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_names


def build_preprocessor(feature_names: list) -> ColumnTransformer:
    """Cree un preprocessor sklearn avec StandardScaler sur toutes les features."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, list(range(len(feature_names))))]
    )
