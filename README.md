# MLOps Diabetes Classification Project

## Contexte

Ce projet transforme un notebook Jupyter expérimental en un pipeline ML reproductible et traçable avec MLflow pour la prédiction du diabète de type 2.

## Problème

- **Médical**: Prédire si un patient est diabétique à partir de mesures cliniques (glycémie, IMC, âge, etc.)
- **MLOps**: Créer un pipeline structuré, reproductible et traçable

## Dataset

**Pima Indians Diabetes** (768 patientes femmes):
- 8 features cliniques
- 35% diabétiques (classe 1), 65% sains (classe 0)
- Déséquilibre de classes à prendre en compte

## Structure du Projet

```
tp_diabetes_mlops/
├── config.py              # Configuration centrale
├── requirements.txt       # Dépendances
├── src/
│   ├── data_loader.py     # Chargement & preprocessing
│   ├── evaluate.py        # Métriques & visualisations
│   └── train.py           # Entraînement avec MLflow
├── models/                # Modèles & artefacts sauvegardés
├── mlruns/                # Expériences MLflow
├── run_experiments.py     # 4 runs RandomForest
├── run_optimized.py       # Optimisation GradientBoosting
├── run_xgboost.py         # XGBoost v1
└── run_xgboost_v2.py      # XGBoost v2
```

## Installation

```bash
# Créer l'environnement virtuel
python -m venv tp_mlops
source tp_mlops/bin/activate  # Linux/macOS
tp_mlops\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Pour XGBoost
pip install xgboost
```

## Utilisation

### Lancer les expériences de base (4 RandomForest)
```bash
python run_experiments.py
```

### Lancer l'optimisation XGBoost
```bash
python run_xgboost_v2.py
```

### Démarrer l'interface MLflow
```bash
mlflow ui
```
Accédez à: http://127.0.0.1:5000

## Résultats

| Modèle | ROC-AUC | Recall | F1 |
|--------|---------|--------|-----|
| RF_baseline | 0.8147 | 0.5926 | 0.6337 |
| RF_balanced | 0.8150 | 0.6296 | 0.6296 |
| RF_large_balanced | 0.8228 | 0.6852 | 0.6549 |
| **XGB_v2_2** | **0.8281** | **0.8704** | **0.7176** |

### Meilleur modèle: XGB_v2_2
- ROC-AUC: 0.8281
- Recall: 87.04% (détecte 87% des diabétiques)
- F1-Score: 0.7176

## Fonctionnalités Implémentées

✅ Structure professionnelle (séparation code/données/modèles)  
✅ Conversion notebook → scripts Python modulaires  
✅ Pipeline scikit-learn (preprocessing + classification)  
✅ MLflow Tracking (params, metrics, artefacts)  
✅ Comparaison des expériences dans l'interface MLflow  
✅ Modèles enregistrés dans MLflow Model Registry  
✅ Validation croisée 5-fold  
✅ Feature Importance sauvegardée  
✅ Courbes ROC et matrices de confusion  
✅ Optimisation du seuil de décision  
✅ XGBoost avec scale_pos_weight  

## Défis Atteints

- **Defi 1**: GridSearchCV avec MLflow ✅
- **Defi 2**: GradientBoosting vs RandomForest ✅
- **Defi 3**: Feature Importance ✅
- **Defi 4**: Validation Croisée 5-fold ✅
- **Defi 5**: Seuil de décision optimal ✅
- **Defi 6**: Courbe ROC ✅

## Technologies

- Python 3.12
- scikit-learn (RandomForest, GradientBoosting)
- XGBoost
- MLflow 2.x
- pandas, numpy
- matplotlib, seaborn

## Bonnes Pratiques MLOps

1. **Reproductibilité**: Configuration centralisée (config.py), random_state fixe
2. **Traçabilité**: Chaque run MLflow enregistre params, metrics, artefacts
3. **Modularité**: Séparation data_loader, evaluate, train
4. **Pipeline complet**: Preprocessing + Classification dans un Pipeline sklearn

## Avertissement

Ce modèle est à des fins éducatives. Pour un usage médical réel, des validations supplémentaires et conformités réglementaires sont nécessaires.
