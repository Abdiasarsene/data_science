# Importation des librairies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Importation de la base de données
base = pd.read_excel(r"d:\Projets\Projet Informatique\Bases de données\canada_data_cleaned (1).xlsx")

# Normalisation de la base de données
base = StandardScaler().fit_transform(base)

# l'ACP
acp = PCA(n_components=7)
pca = acp.fit_transform(base)

# Création de la nouvelle base de données après l'analyse en composante principale
pca_data = pd.DataFrame(data = pca, columns=("cible","predicteur_1", "predicteur_2","predicteur_3","contrôle_1", "contrôle_2", "contrôle_3" ))

pca_data # Affichae

# Visualisation des données après ACP
sns.pairplot(pca_data)
plt.show()

# Interprétation des résultats
explained_variance = acp.explained_variance_ratio_
print("cible : ", explained_variance[0])
print("predicteur_1 : ", explained_variance[1])
print("predicteur_2 : ", explained_variance[2])
print("predicteur_3 : ", explained_variance[3])
print("contrôle_1 : ", explained_variance[4])
print("contrôle_2 : ", explained_variance[5])
print("contrôle_3 : ", explained_variance[6])

# Préparation des variables
x = pca_data.drop(columns=['cible']) #Prédicteurs;
y = pca_data['cible'] #cible

# Dvision des données en enseomble d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=808)

# Entraînement du model
model =LinearRegression().fit(x_train, y_train)

# Prédiction 
y_pred = model.predict(x_test)

# Evaluation de la capacité prédictive
print("MAE : ", mean_absolute_error(y_test, y_pred))
print("Regression linéaire MAPE : ", mean_absolute_percentage_error(y_test, y_pred))
print("RMSE : ", root_mean_squared_error(y_test, y_pred))

"""Test avec d'autres modèles comme les méthodes régularisées et l'ensemble learning"""

# Entraînement du modèle de la forêt aléatoire
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

y_pred_rf_model = rf_model.predict(x_test) #Prédiction

# Evaluation de la capacité prédictive
print(" Random Forest MAPE : ", mean_absolute_percentage_error(y_test, y_pred_rf_model))

"""Méthode régularisée  Ridge  & Lasso"""
rid_model = Ridge(alpha=1.0)
las_model = Lasso(alpha=0.1)
rid_model.fit(x_train, y_train)
las_model.fit(x_train, y_train)
y_pred_rid = rid_model.predict(x_test)
y_pred_las = las_model.predict(x_test)

# Evaluation de la capacité prédictive
print(" Ridge MAPE : ", mean_absolute_percentage_error(y_test, y_pred_rid))
print(" Lasso MAPE : ", mean_absolute_percentage_error(y_test, y_pred_las))

"""Le modèle 'Random Forest' souffre-t-il d'overfitting"""
# Définir les paramètres à tester
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialisation du modèle
rf_model = RandomForestRegressor(random_state=42)

# Grid Search
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# Meilleurs paramètres
print(f"Best parameters: {grid_search.best_params_}")

# Prédictions avec le meilleur modèle
best_rf_model = grid_search.best_estimator_
y_pred_best_rf = best_rf_model.predict(x_test)

# Évaluation
mape_best_rf = mean_absolute_percentage_error(y_test, y_pred_best_rf)
print(f"Optimized Random Forest MAE: {mape_best_rf}")
