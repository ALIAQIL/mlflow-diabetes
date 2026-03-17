import os

# Chemins du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Parametres donnees
TARGET_COLUMN = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Parametres MLflow
MLFLOW_EXPERIMENT = "diabetes_classification"
MODEL_NAME = "diabetes_random_forest"

# Creer les dossiers s'ils n'existent pas
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
