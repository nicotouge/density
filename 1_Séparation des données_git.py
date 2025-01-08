# import os
# os.chdir("c:\users\pierr\desktop\cours estp\ipgp\système volcanique\tutored project")

# Libraries necessary to do the job (non exhaustive)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score

## Code portability (to change)
from pathlib import Path

# Définir le chemin racine #Chemin à changer
racine = Path(r"C:\Users\")

# IMPORT THE DATA
db = pd.read_csv(racine / "SCIGLASS_DATASET.csv")
print(db.head())
print("Number of data points: {}".format(len(db)))

# we record the elements we have in a list
# easier after when asking things to Pandas tables
elements_ = ['sio2', 'tio2', 'al2o3', 'feo', 'fe2o3',
             'mno', 'na2o', 'k2o', 'mgo', 'cao', 'p2o5',
             'h2o']

# TRAIN-TEST SPLIT
#
train, test = train_test_split(db, test_size=0.2)

# GET X and Y
# we also put everything as numpy arrays
X_train, y_train = train.loc[:, elements_].values, train.loc[:,"d"].values
X_test, y_test = test.loc[:, elements_].values, test.loc[:,"d"].values

df1 = pd.DataFrame(X_train)
df2 = pd.DataFrame(y_train)
df3 = pd.DataFrame(X_test)
df4 = pd.DataFrame(y_test)

df1.to_csv(racine / "X_train.csv", index=False)
df2.to_csv(racine / "y_train.csv", index=False)
df3.to_csv(racine / "X_test.csv", index=False)
df4.to_csv(racine / "y_test.csv", index=False)

# STANDARDIZATION
# you could also try a standard scaler
X_scaler = StandardScaler().fit(X_train)
X_scaler2 = MinMaxScaler().fit(X_train)

# scaling the datasets (suffix _sc)
X_train_sc = X_scaler.transform(X_train)
X_test_sc = X_scaler.transform(X_test)

X_train_sc2 = X_scaler2.transform(X_train)
X_test_sc2 = X_scaler2.transform(X_test)

with open(racine / "X_train_sc.pkl", "wb") as file:
    pickle.dump(X_train_sc, file)

with open(racine / "X_test_sc.pkl", "wb") as file:
    pickle.dump(X_test_sc, file)

with open(racine / "X_train_sc2.pkl", "wb") as file:
    pickle.dump(X_train_sc2, file)

with open(racine / "X_test_sc2.pkl", "wb") as file:
    pickle.dump(X_test_sc2, file)