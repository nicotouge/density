#Attention code long à exécuter +- 10 min
## Libraries necessary to do the job (non exhaustive)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

elements_ = ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3',
             'mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5',
             'h2o']

## Portabilité du code (à changer !!!)
from pathlib import Path

# Définir le chemin racine #Chemin à changer
racine = Path(r"C:\Users\")

## Importation des données stockées & modèles de régression choisis
# Chargement du fichier CSV
db = pd.read_csv(racine / "SCIGLASS_DATASET.csv")

X_train=pd.read_csv(racine / "X_train.csv")
y_train=pd.read_csv(racine / "y_train.csv")
X_test=pd.read_csv(racine / "X_test.csv")
y_test=pd.read_csv(racine / "y_test.csv")

with open(racine / "X_train_sc.pkl", "rb") as file:
    X_train_sc = pickle.load(file)

with open(racine / "X_test_sc.pkl", "rb") as file:
    X_test_sc = pickle.load(file)

with open(racine / "X_train_sc2.pkl", "rb") as file:
    X_train_sc2 = pickle.load(file)

with open(racine / "X_test_sc2.pkl", "rb") as file:
    X_test_sc2 = pickle.load(file)

y_train = y_train.values.ravel()
y_test = y_test.values


##Optimisation des paramètres de la modélisation kernel (Optimisation bayésienne)
# Définir les hyperparamètres à tester
# def objective(trial):
#     hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)])
#     alpha = trial.suggest_loguniform('alpha', 1e-4, 1e-1)
#     learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive']

# def objective(trial):
#     hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(10, 5), (20, 10), (10,10), (20,20), (30,20)])
#     alpha = trial.suggest_loguniform('alpha', 1e-4, 1e-1)
#     learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive']))

def objective(trial):
    hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(20,20), (20,30), (30,30), (50,50), (50, 100), (100, 50)])
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1e-1)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])


# Modèle avec les hyperparamètres suggérés
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=1000,
        random_state=42
    )

    # Validation croisée
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()  # Minimise l'erreur quadratique moyenne

# Optimisation
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Résultats
print("Best parameters:", study.best_params)
print("Best score:", study.best_value)

#Best parameters: {'hidden_layer_sizes': (10, 5), 'alpha': 0.09819495428084557, 'learning_rate': 'adaptive'}, Best score: 0.011221882174607045
#Best parameters: {'hidden_layer_sizes': (10, 5), 'alpha': 0.09322145116868696, 'learning_rate': 'constant'}, Best score: 0.011233522295282806
#Best parameters: {'hidden_layer_sizes': (20, 20), 'alpha': 0.0934940372311712, 'learning_rate': 'constant'}, Best score: 0.009656530865418638
#Best parameters: {'hidden_layer_sizes': (30, 30), 'alpha': 0.09806722722382458, 'learning_rate': 'constant'}, Best score: 0.008745941226795674