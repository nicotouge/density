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
from sklearn.metrics import median_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_validate
import xgboost as xgb
from mapie.regression import MapieRegressor, MapieQuantileRegressor
from sklearn.base import BaseEstimator
import copy

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


## Training different regression models
#Modèle linéaire
linear_m = linear_model.LinearRegression() #Normal scaler
linear_m.fit(X_train_sc, y_train.ravel())

linear_m2 = linear_model.LinearRegression() #MinMax Scaler
linear_m2.fit(X_train_sc2, y_train.ravel())

#Modèle linéaire élastique
elastic_m=linear_model.ElasticNetCV(l1_ratio=0.1) #pas d'influence de la variation du l1_ration ou d'epsilon
elastic_m.fit(X_train_sc, y_train.ravel())

#Modèle kernel ridge regression
kernel_ridge_m=KernelRidge(alpha=0.0027325480535122498, kernel='rbf', gamma=0.010074240715006027) #Normal Scaler
kernel_ridge_m.fit(X_train_sc, y_train.ravel())
#Meilleurs hyperparamètres :  {'alpha': 0.0027325480535122498, 'gamma': 0.010074240715006027}

kernel_ridge_m2=KernelRidge(alpha=0.002790523938022217, kernel='rbf', gamma=0.010058006061503259) #MinMax Scaler
kernel_ridge_m2.fit(X_train_sc2, y_train.ravel())
#Meilleurs hyperparamètres :  {'alpha': 0.002790523938022217, 'gamma': 0.010058006061503259}

MLPR_m = MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.09806722722382458, learning_rate='constant', max_iter=1000, random_state=42) #Normal scaler only
MLPR_m.fit(X_train_sc, y_train.ravel())
#Best parameters: {'hidden_layer_sizes': (30, 30), 'alpha': 0.09806722722382458, 'learning_rate': 'constant'}, Best score: 0.008745941226795674

XGB_m=xgb.XGBRegressor(n_estimators=320, max_depth=10, learning_rate=0.041220225453779055, subsample=0.6787377271729647, colsample_bytree=0.8076480841795373, gamma=0.0008085030734184188, reg_alpha=1.6326087745207382e-07, reg_lambda=1.0395264401495856)
XGB_m.fit(X_train_sc, y_train.ravel())
#Best parameters: {'n_estimators': 320, 'max_depth': 10, 'learning_rate': 0.041220225453779055, 'subsample': 0.6787377271729647, 'colsample_bytree': 0.8076480841795373, 'gamma': 0.0008085030734184188, 'reg_alpha': 1.6326087745207382e-07, 'reg_lambda': 1.0395264401495856}, Best MSE: 0.004491715422817194

with open(racine / 'xgboost_model.pkl', 'wb') as file:
    pickle.dump(XGB_m, file)

with open(racine / 'ANN_model.pkl', 'wb') as file:
    pickle.dump(MLPR_m, file)

## Evaluation 1
rmse_train1 = root_mean_squared_error(linear_m.predict(X_train_sc), y_train) #Attention fonction dans les versions ultérieures de SKLearn
rmse_test1 = root_mean_squared_error(linear_m.predict(X_test_sc), y_test)

rmse_train12 = root_mean_squared_error(linear_m2.predict(X_train_sc2), y_train)
rmse_test12 = root_mean_squared_error(linear_m2.predict(X_test_sc2), y_test)

rmse_train2 = root_mean_squared_error(elastic_m.predict(X_train_sc), y_train)
rmse_test2 = root_mean_squared_error(elastic_m.predict(X_test_sc), y_test)

rmse_train3 = root_mean_squared_error(kernel_ridge_m.predict(X_train_sc), y_train)
rmse_test3 = root_mean_squared_error(kernel_ridge_m.predict(X_test_sc), y_test)

rmse_train32 = root_mean_squared_error(kernel_ridge_m2.predict(X_train_sc2), y_train)
rmse_test32 = root_mean_squared_error(kernel_ridge_m2.predict(X_test_sc2), y_test)

rmse_train4 = root_mean_squared_error(MLPR_m.predict(X_train_sc), y_train)
rmse_test4 = root_mean_squared_error(MLPR_m.predict(X_test_sc), y_test)

rmse_train5 = root_mean_squared_error(XGB_m.predict(X_train_sc), y_train)
rmse_test5 = root_mean_squared_error(XGB_m.predict(X_test_sc), y_test)

## Evaluation 1 cross validation
# Scorer pour RMSE (Scikit-learn utilise les scorers où plus c'est grand, mieux c'est)
scorer1 = make_scorer(root_mean_squared_error, greater_is_better=False)

rmse_scores1 = cross_validate(linear_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer1, return_train_score=True)
rmse_scores11 = cross_validate(linear_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer1)
rmse_train_c1=-rmse_scores1['train_score'].mean()
rmse_validate_c1=-rmse_scores1['test_score'].mean()
rmse_test_c1=-rmse_scores11['test_score'].mean()

rmse_scores2 = cross_validate(kernel_ridge_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer1, return_train_score=True)
rmse_scores21 = cross_val_score(kernel_ridge_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer1)
rmse_train_c2=-rmse_scores2['train_score'].mean()
rmse_validate_c2=-rmse_scores2['test_score'].mean()
rmse_test_c2=-np.mean(rmse_scores21)

rmse_scores3 = cross_validate(MLPR_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer1, return_train_score=True)
rmse_scores31 = cross_val_score(MLPR_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer1)
rmse_train_c3=-rmse_scores3['train_score'].mean()
rmse_validate_c3=-rmse_scores3['test_score'].mean()
rmse_test_c3=-np.mean(rmse_scores31)

# rmse_scores4 = cross_validate(XGB_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer1, return_train_score=True)
# rmse_scores41 = cross_val_score(XGB_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer1)
# rmse_train_c4=-rmse_scores4['train_score'].mean()
# rmse_validate_c4=-rmse_scores4['test_score'].mean()
# rmse_test_c4=-np.mean(rmse_scores41)


## Evaluation 2
r2_train1 = r2_score(linear_m.predict(X_train_sc), y_train)
r2_test1 = r2_score(linear_m.predict(X_test_sc), y_test)

r2_train12 = r2_score(linear_m2.predict(X_train_sc2), y_train)
r2_test12 = r2_score(linear_m2.predict(X_test_sc2), y_test)

r2_train2 = r2_score(elastic_m.predict(X_train_sc), y_train)
r2_test2 = r2_score(elastic_m.predict(X_test_sc), y_test)

r2_train3 = r2_score(kernel_ridge_m.predict(X_train_sc), y_train)
r2_test3 = r2_score(kernel_ridge_m.predict(X_test_sc), y_test)

r2_train32 = r2_score(kernel_ridge_m2.predict(X_train_sc2), y_train)
r2_test32 = r2_score(kernel_ridge_m2.predict(X_test_sc2), y_test)

r2_train4 = r2_score(MLPR_m.predict(X_train_sc), y_train)
r2_test4 = r2_score(MLPR_m.predict(X_test_sc), y_test)

r2_train5 = r2_score(XGB_m.predict(X_train_sc), y_train)
r2_test5 = r2_score(XGB_m.predict(X_test_sc), y_test)

## Evaluation 2 cross validation
# Scorer pour RMSE (Scikit-learn utilise les scorers où plus c'est grand, mieux c'est)
scorer2 = make_scorer(r2_score, greater_is_better=True)

r2_scores1 = cross_validate(linear_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer2, return_train_score=True)
r2_scores11 = cross_val_score(linear_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer2)
r2_train_c1=r2_scores1['train_score'].mean()
r2_validate_c1=r2_scores1['test_score'].mean()
r2_test_c1=np.mean(r2_scores11)

r2_scores2 = cross_validate(kernel_ridge_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer2, return_train_score=True)
r2_scores21 = cross_val_score(kernel_ridge_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer2)
r2_train_c2=r2_scores2['train_score'].mean()
r2_validate_c2=r2_scores2['test_score'].mean()
r2_test_c2=np.mean(r2_scores21)

r2_scores3 = cross_validate(MLPR_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer2, return_train_score=True)
r2_scores31 = cross_val_score(MLPR_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer2)
r2_train_c3=r2_scores3['train_score'].mean()
r2_validate_c3=r2_scores3['test_score'].mean()
r2_test_c3=np.mean(r2_scores31)

r2_scores4 = cross_validate(XGB_m, X_train_sc, y_train.ravel(), cv=5, scoring=scorer2, return_train_score=True)
r2_scores41 = cross_val_score(XGB_m, X_test_sc, y_test.ravel(), cv=2, scoring=scorer2)
r2_train_c4=r2_scores4['train_score'].mean()
r2_validate_c4=r2_scores4['test_score'].mean()
r2_test_c4=np.mean(r2_scores41)

## Conformal prediction - Mapie
# Convertir y_train et y_test en vecteurs unidimensionnels
y_train = y_train.ravel()
y_test = y_test.ravel()

# Appliquer Mapie pour les intervalles
#kernel_ridge_m #MLPR_m #XGB_m
# mapie1 = MapieRegressor(estimator=linear_m, method="naive")
# mapie1.fit(X_train, y_train)
#
# mapie12 = MapieQuantileRegressor(estimator=linear_m)
# mapie12.fit(X_train, y_train)
#
# # mapie2 = MapieRegressor(estimator=kernel_ridge_m, method="plus") #RAM insuffisante
# # mapie2.fit(X_train, y_train)
#
# # mapie22 = MapieQuantileRegressor(estimator=kernel_ridge_m)
# # mapie22.fit(X_train, y_train)
#
mapie3 = MapieRegressor(estimator=MLPR_m, method="plus")
mapie3.fit(X_train, y_train)

with open(racine / 'mapie_ANN_model.pkl', 'wb') as file:
    pickle.dump(mapie3, file)

# mapie32 = MapieQuantileRegressor(estimator=MLPR_m)
# mapie32.fit(X_train, y_train)

mapie4 = MapieRegressor(estimator=XGB_m, method="plus")
mapie4.fit(X_train, y_train)

with open(racine / 'mapie_xgboost_model.pkl', 'wb') as file:
    pickle.dump(mapie4, file)

# mapie42 = MapieQuantileRegressor(estimator=XGB_m)
# mapie42.fit(X_train, y_train)


# Calculer les intervalles pour 2 sigma (95%)
y_pred1, y_pis_2sigma1 = mapie1.predict(X_test, alpha=0.05)       # 95% confidence interval
#y_pred2, y_pis_2sigma2 = mapie2.predict(X_test, alpha=0.05)
y_pred3, y_pis_2sigma3 = mapie3.predict(X_test, alpha=0.05)
y_pred4, y_pis_2sigma4 = mapie4.predict(X_test, alpha=0.05)

rmse1=root_mean_squared_error(y_test,y_pred1)
rmse3=root_mean_squared_error(y_test,y_pred3)
rmse4=root_mean_squared_error(y_test,y_pred4)

r2_1=r2_score(y_test,y_pred1)
r2_3=r2_score(y_test,y_pred3)
r2_4=r2_score(y_test,y_pred4)

# Résultats simplifiés
# results = pd.DataFrame({
#     "y_test": y_test,        # Valeurs réelles
#     "y_pred": y_pred1,        # Valeurs prédites
#     "PI_lower_95": y_pis_2sigma1[:,0, 0], # Borne inférieure pour 95%
#     "PI_upper_95": y_pis_2sigma1[:, 1,0], # Borne supérieure pour 95%
# })

sigma1_mean=len(y_test)*[np.mean(y_pis_2sigma1[:, 1, 0] - y_pis_2sigma1[:, 0, 0])]
#sigma2_mean=len(y_test)*[np.mean(y_pis_2sigma2[:, 1, 0] - y_pis_2sigma2[:, 0, 0])]
sigma3_mean=len(y_test)*[np.mean(y_pis_2sigma3[:, 1, 0] - y_pis_2sigma3[:, 0, 0])]
sigma4_mean=len(y_test)*[np.mean(y_pis_2sigma4[:, 1, 0] - y_pis_2sigma4[:, 0, 0])]

plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pis_2sigma1[:, 1, 0]-y_pis_2sigma1[:, 0,0], 'o', label="Modélisation linéaire", color="blue")
plt.plot(y_test, sigma1_mean, '--', color="blue")
#plt.plot(y_test, y_pis_2sigma2[:, 1, 0]-y_pis_2sigma2[:, 0,0], 'o', label="Modélisation Kernel Ridge", color="red")
#plt.plot(y_test, sigma2_mean, '--', color="red")
plt.plot(y_test, y_pis_2sigma3[:, 1, 0]-y_pis_2sigma3[:, 0,0], 'o', label="Modélisation ANN", color="green")
plt.plot(y_test, sigma3_mean, '--', color="green")
plt.plot(y_test, y_pis_2sigma4[:, 1, 0]-y_pis_2sigma4[:, 0,0], 'o', label="Modélisation XGBoost", color="black")
plt.plot(y_test, sigma4_mean, '--', color="black")

plt.xlabel("Densité")
plt.ylabel("Largeur de l'intervalle de prédiction")
plt.legend()
plt.title("Conformal Prediction avec MAPIE")
plt.show()

## Affichage des résultats
# print("\n### Performances without the cross-validation ###")
# print("\n### Linear model performance metrics ###")
# print("## Normal Standardisation ##")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train1, r2_train1))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test1, r2_test1))
# print("## MinMax Standardisation ##")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train12, r2_train12))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test12, r2_test12))
#
# print("\n### Elastic Net CV model performance metrics (Normal Standardisation only) ###")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train2, r2_train2))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test2, r2_test2))
#
# print("\n### Kernel Ridge model performance metrics ###")
# print("## Normal Standardisation ##")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train3, r2_train3))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test3, r2_test3))
# print("## MinMax Standardisation ##")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train32, r2_train32))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test32, r2_test32))
#
# print("\n### Artificial neural network model performance metrics (Normal Stdr only) ###")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train4, r2_train4))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test4, r2_test4))

print("\n\n### Performances with the cross-validation ###")
print("\n### Linear model performance metrics ###")
print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train_c1, r2_train_c1))
print("Validation data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_validate_c1, r2_validate_c1))
print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test1, r2_test1))
print("\n### Kernel Ridge model performance metrics ###")
print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train_c2, r2_train_c2))
print("Validation data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_validate_c2, r2_validate_c2))
print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test2, r2_test2))
print("\n### Artificial neural network model performance metrics  ###")
print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train_c3, r2_train_c3))
print("Validation data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_validate_c3, r2_validate_c3))
print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test4, r2_test4))
# print("\n### XGBoost model performance metrics  ###")
# print("Training data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_train_c4, r2_train_c4))
# print("Validation data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_validate_c4, r2_validate_c4))
# print("Testing data subset, RMSE {:.2f}, R2 {:.2f}".format(rmse_test5, r2_test5))

print("\n### Conformal prediction with Mapie ###")
print("Linear model, RMSE {:.2f}, R2 {:.2f}".format(rmse1, r2_1))
print("ANN model, RMSE {:.2f}, R2 {:.2f}".format(rmse3, r2_3))
print("XGBoost model, RMSE {:.2f}, R2 {:.2f}".format(rmse4, r2_4))