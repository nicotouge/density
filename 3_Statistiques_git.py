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
import seaborn as sns


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

y_train = y_train.values
y_test = y_test.values

## Matrice de corrélation de Pearson
# Extraction des données X et y
X = pd.DataFrame(db.loc[:, elements_].values, columns=elements_)  # Conversion en DataFrame
y = pd.Series(db.loc[:, "d"].values, name="d")  # Conversion en Series

# Concaténation des données
df = pd.concat([X, y], axis=1)

# Calcul de la corrélation de Pearson
pear_corr = df.corr(method='pearson')

# Application de style pour l'affichage
pear_corr.style.background_gradient(cmap='Greens', axis=0)

plt.figure(figsize=(10, 8))  # Taille de la figure
sns.heatmap(pear_corr, annot=True, fmt=".2f", cmap="Greens", cbar=True, square=True)

# Ajout des titres
plt.title("Correlation matrix (Pearson)", fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotation des labels sur l'axe x
plt.yticks(rotation=0)               # Pas de rotation pour les labels de l'axe y
plt.tight_layout()                   # Ajustement automatique pour éviter que les étiquettes se chevauchent

# Affichage de la figure
plt.show()

## Statistiques
# Tracer les histogrammes pour toutes les colonnes numériques

nombre = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 3 lignes, 4 colonnes

for idx, (col, ax, nb) in enumerate(zip(elements_, axes.ravel(), nombre)):
    # Utiliser 'idx' comme nom de colonne pour X_train
    column_name = str(idx)  # Convertir l'indice en chaîne pour accéder à la colonne
    ax.hist(X_train[column_name], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f"{col}")  # Titre avec le nom de l'élément et 'nombre'
    ax.set_xlabel("Normalized content (in %)")
    ax.set_ylabel("Frequency")

fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # 3 lignes, 4 colonnes

for idx, (col, ax, nb) in enumerate(zip(elements_, axes.ravel(), nombre)):
    # Utiliser 'idx' comme nom de colonne pour X_train
    column_name = str(idx)  # Convertir l'indice en chaîne pour accéder à la colonne
    ax.hist(X_test[column_name], bins=20, color='orange', edgecolor='black')
    ax.set_title(f"{col}")  # Titre avec le nom de l'élément et 'nombre'
    ax.set_xlabel("Normalized content (in %)")
    ax.set_ylabel("Frequency")


fig, ax= plt.subplots()
ax= sns.boxplot(data=y_train, palette="Set3")
ax.set_ylabel("Density")
ax.set_title("Box-and-whisker plot depicting the distribution of training densities")

fig, ax= plt.subplots()
ax= sns.boxplot(data=y_test, palette="Set3")
ax.set_ylabel("Densité")
ax.set_title("Box-and-whisker plot depicting the distribution of testing densities")


plt.show()