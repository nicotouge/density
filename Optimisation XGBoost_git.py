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
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_validate
import xgboost as xgb

elements_ = ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3',
             'mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5',
             'h2o']
## Portabilité du code (à changer !!!)
from pathlib import Path

# Définir le chemin racine #Chemin à changer
racine = Path(r"C:\Users\")

## Importation des données stockées & modèles de régression choisis
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

y_train = y_train.values
y_test = y_test.values

##Optimisation des paramètres de la modélisation XGBoost (Optimisation bayésienne)
#Diviser l'ensemble d'entraînement en sous-ensembles : entraînement et validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Fonction objective pour Optuna
def objective(trial):
    # Définir les hyperparamètres à optimiser
    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",  # Choisir le booster de base pour la régression
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_uniform("gamma", 0, 5),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10),
    }

    # Convertir les données en tableaux NumPy
    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32)
    X_val_np = np.array(X_val, dtype=np.float32)
    y_val_np = np.array(y_val, dtype=np.float32)

    # Créer les DMatrix
    dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    dval = xgb.DMatrix(X_val_np, label=y_val_np)

    # Entraîner le modèle avec arrêt anticipé
    model = xgb.train(params, dtrain, evals=[(dval, 'eval')], num_boost_round=params['n_estimators'],
                      early_stopping_rounds=10, verbose_eval=False)

    # Faire des prédictions
    preds = model.predict(dval)

    # Calculer l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(y_val, preds)
    return mse


# Créer l'objet Study
study = optuna.create_study(direction="minimize")
# Exécuter l'optimisation
study.optimize(objective, n_trials=500)

# Afficher les résultats
print("Best parameters:", study.best_params)
print("Best MSE:", study.best_value)

#Best parameters: {'n_estimators': 246, 'max_depth': 7, 'learning_rate': 0.09491417456729673, 'subsample': 0.9120656204821478, 'colsample_bytree': 0.6787846416957366, 'gamma': 0.08068495313487789, 'reg_alpha': 7.081316104300238e-06, 'reg_lambda': 6.070625238222313e-06}, Best MSE: 0.0054965474505972636
#Best parameters: {'n_estimators': 101, 'max_depth': 8, 'learning_rate': 0.10487178442624158, 'subsample': 0.9999052467968682, 'colsample_bytree': 0.7634183940824802, 'gamma': 0.03665363075877179, 'reg_alpha': 7.738730709987554e-07, 'reg_lambda': 0.00015976610586130837}, Best MSE: 0.005271400586558446
#Best parameters: {'n_estimators': 320, 'max_depth': 10, 'learning_rate': 0.041220225453779055, 'subsample': 0.6787377271729647, 'colsample_bytree': 0.8076480841795373, 'gamma': 0.0008085030734184188, 'reg_alpha': 1.6326087745207382e-07, 'reg_lambda': 1.0395264401495856}, Best MSE: 0.004491715422817194