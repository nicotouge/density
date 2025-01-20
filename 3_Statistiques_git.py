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
elements_1 = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'Fe2O3',
             'MnO', 'Na2O', 'K2O', 'MgO', 'CaO', 'P2O5',
             'H2O']

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

# Remplace les valeurs de la diagonale par NaN
np.fill_diagonal(pear_corr.values, np.nan)

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
fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)  # 3 lignes, 4 colonnes

for idx, (col, ax, nb) in enumerate(zip(elements_, axes.ravel(), nombre)):
    # Utiliser 'idx' comme nom de colonne pour X_train
    column_name = str(idx)  # Convertir l'indice en chaîne pour accéder à la colonne
    ax.hist(X_train[column_name], bins=20, color='skyblue', edgecolor='black')
    nom=elements_1[idx]
    col_with_indices = nom.replace("2", "_{2}").replace("3", "_{3}").replace("5", "_{5}")
    ax.set_title(f"${col_with_indices}$")  # Ajouter le format LaTeX
    ax.set_xlabel("Normalized content (in %)")
    ax.set_ylabel("Frequency")

fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)  # 3 lignes, 4 colonnes

for idx, (col, ax, nb) in enumerate(zip(elements_, axes.ravel(), nombre)):
    # Utiliser 'idx' comme nom de colonne pour X_train
    column_name = str(idx)  # Convertir l'indice en chaîne pour accéder à la colonne
    ax.hist(X_test[column_name], bins=20, color='orange', edgecolor='black')
    ax.set_title(f"{col}")  # Titre avec le nom de l'élément et 'nombre'
    ax.set_xlabel("Normalized content (in %)")
    ax.set_ylabel("Frequency")

for idx, (col, ax, nb) in enumerate(zip(elements_, axes.ravel(), nombre)):
    # Utiliser 'idx' comme nom de colonne pour X_train
    column_name = str(idx)  # Convertir l'indice en chaîne pour accéder à la colonne
    ax.hist(X_test[column_name], bins=20, color='orange', edgecolor='black')

    # Formater le titre avec LaTeX pour les indices
    nom=elements_1[idx]
    col_with_indices = nom.replace("2", "_{2}").replace("3", "_{3}").replace("5", "_{5}")
    ax.set_title(f"${col_with_indices}$")  # Ajouter le format LaTeX

    ax.set_xlabel("Normalized content (in %)")
    ax.set_ylabel("Frequency")

mean_value1 = np.mean(y_train)
median_value1 = np.median(y_train)
mean_value2 = np.mean(y_test)
median_value2 = np.median(y_test)

# Définir les limites horizontales communes
min_value = min(min(y_train), min(y_test))
max_value = max(max(y_train), max(y_test))



fig, axes = plt.subplots(2, 1, figsize=(8, 6),constrained_layout=True)
axes[0].hist(y_train, bins=30, color='skyblue', edgecolor='black')
axes[0].axvline(mean_value1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value1:.2f}')
axes[0].axvline(mean_value1, color='red', linestyle='--', linewidth=2, label=f'Median: {median_value1:.2f}')
axes[0].set_xlim(min_value, max_value)
axes[0].set_xlabel("Density (in g/$cm_{3}$)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram of the density distribution in the training dataset")
axes[0].legend(fontsize=10)


axes[1].hist(y_test, bins=30, color='orange', edgecolor='black')
axes[1].axvline(mean_value2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value2:.2f}')
axes[1].axvline(mean_value2, color='red', linestyle='--', linewidth=2, label=f'Median: {median_value2:.2f}')
axes[1].set_xlim(min_value, max_value)
axes[1].set_xlabel("Density (in g/$cm_{3}$)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Histogram of the density distribution in the test dataset")
axes[1].legend(fontsize=10)


plt.show()