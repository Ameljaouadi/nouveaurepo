import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, precision_score, f1_score

import mlflow
import mlflow.sklearn
from datetime import datetime

# Définir l'URL de suivi MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Générer un nom d'expérimentation dynamique basé sur la date et l'heure
experiment_name = f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Définir l'expérimentation dynamique
mlflow.set_experiment(experiment_name)

# Charger les données

reference_data = pd.read_csv('../data/reference_data.csv')
new_data = pd.read_csv('../data/new_data.csv')

# Combiner les données
df= pd.concat([reference_data, new_data], ignore_index=True)

# Séparer les caractéristiques et la cible

X = df.drop('Outcome', axis=1)
y = df['Outcome']
# Diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement des caractéristiques numériques

numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Transformer les colonnes

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
# Créer le pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
# Démarrer une nouvelle expérimentation MLflow

mlflow.start_run()
run_id = mlflow.active_run().info.run_id
print("Run ID:", run_id)

# Loguer des paramètres et des métriques
mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("data_source", "reference_data + new_data")
# Entraîner le modèle

pipeline.fit(X_train, y_train)
# Prédictions sur l'ensemble de test
y_pred = pipeline.predict(X_test)

# Calculer les métriques
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Loguer les métriques dans MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("f1_score", f1)

# Enregistrer le modèle avec MLflow
mlflow.sklearn.log_model(pipeline, "model")

# Sauvegarder le modèle localement avec pickle (facultatif si vous voulez aussi un fichier pickle)

with open('../models/pipeline4.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Sélectionner le meilleur modèle selon les métriques
# On récupère tous les runs dans l'expérimentation en cours
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id])



# Trouver le run avec le meilleur score F1 (par exemple)
#best_run = runs.loc[runs['f1_score'].idxmax()]
best_run = runs.loc[runs['metrics.f1_score'].idxmax()]

best_model_uri = f"runs:/{best_run.run_id}/model"
best_model = mlflow.sklearn.load_model(best_model_uri)

#best_model_dir = '../mymodel'
#os.makedirs(best_model_dir, exist_ok=True)

# Sauvegarder le meilleur modèle localement
best_model_path = '../mymodel/best_model.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Meilleur modèle sauvegardé localement dans '{best_model_path}'.")

# Charger le meilleur modèle
#best_model_uri = best_run.artifacts['model']
best_model = mlflow.sklearn.load_model(best_model_uri)

# Utiliser ce modèle pour faire des prédictions futures
y_pred_best_model = best_model.predict(X_test)



# Terminer l'exécution de MLflow
mlflow.end_run()




print(f"Best model (Run ID: {best_run.run_id}) selected for future predictions.")


print("Model training and logging completed.")