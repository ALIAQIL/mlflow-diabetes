import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_and_track

print("=" * 60)
print(" LANCEMENT DES EXPERIENCES MLflow")
print("=" * 60)

results = []

# Experience 1 : Baseline
_, metrics_1, _ = train_and_track(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight=None,
    run_name="RF_baseline",
)
results.append(("RF_baseline", metrics_1))

# Experience 2 : Profondeur limitee (regularisation)
_, metrics_2, _ = train_and_track(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight=None,
    run_name="RF_regularized",
)
results.append(("RF_regularized", metrics_2))

# Experience 3 : Gestion du desequilibre de classes
_, metrics_3, _ = train_and_track(
    n_estimators=150,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight="balanced",
    run_name="RF_balanced",
)
results.append(("RF_balanced", metrics_3))

# Experience 4 : Plus d'estimateurs, feuilles plus grandes
_, metrics_4, _ = train_and_track(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    run_name="RF_large_balanced",
)
results.append(("RF_large_balanced", metrics_4))

# Tableau recapitulatif
print("\n" + "=" * 65)
print(f" {'Run Name':<25} {'ROC-AUC':>8} {'F1':>8} {'Recall':>8} {'Acc.':>8}")
print("=" * 65)
for name, m in results:
    print(
        f" {name:<25} {m['roc_auc']:>8.4f} {m['f1_score']:>8.4f} {m['recall']:>8.4f} {m['accuracy']:>8.4f}"
    )
print("=" * 65)

best = max(results, key=lambda x: x[1]["roc_auc"])
print(f"\nMeilleur modele (ROC-AUC) : {best[0]} -> {best[1]['roc_auc']:.4f}")
print("\nLancez 'mlflow ui' pour visualiser toutes les experiences !")
