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


from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

elements_ = ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3',
             'mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5',
             'h2o']

## Portabilité du code (à changer !!!)
from pathlib import Path

# Définir le chemin racine #Chemin à changer
racine = Path(r"C:\Users\")

## Importation des données stockées & modèles de régression choisis
X_train=pd.read_csv(racine \ "X_train.csv")
y_train=pd.read_csv(racine \ "y_train.csv")
X_test=pd.read_csv(racine \"X_test.csv")
y_test=pd.read_csv(racine \"y_test.csv")


with open(racine \ "X_train_sc2.pkl", "rb") as file:
    X_train_sc = pickle.load(file)

with open(racine \ "X_test_sc2.pkl", "rb") as file:
    X_test_sc = pickle.load(file)

y_train = y_train.values
y_test = y_test.values


##Optimisation des paramètres de la modélisation kernel (Optimisation bayésienne)
def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1e0)
    gamma = trial.suggest_loguniform('gamma', 1e-2, 1e1)

    model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_params = study.best_params

print("Meilleurs hyperparamètres : ", best_params)
#Meilleurs hyperparamètres :  {'alpha': 0.002790523938022217, 'gamma': 0.010058006061503259}