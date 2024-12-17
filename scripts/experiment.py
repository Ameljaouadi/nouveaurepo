import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle


def setup_mlflow():
    """Configure MLflow tracking URI et expérimentation."""
    mlflow.set_tracking_uri("./mlruns")  # Utilise un tracking local
    if not os.path.exists("./mlruns"):
        os.makedirs("./mlruns")
    mlflow.set_experiment("Diabetes Prediction Experiment3")


def log_experiment(data, target, pipeline_path, model_name="DiabetesPipeline"):
    """Enregistre les résultats d'une expérimentation dans MLflow."""
    setup_mlflow()

    # Vérifie l'existence du fichier pipeline
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Le pipeline est introuvable : {pipeline_path}")

    # Charger le pipeline
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)

    # Vérifier que la colonne cible existe
    if target not in data.columns:
        raise ValueError(f"La colonne cible '{target}' n'est pas présente dans les données.")

    # Préparation des données
    X = data.drop(columns=target, axis=1)
    y = data[target]

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prédiction et calcul des métriques
    try:
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="binary", zero_division=0)
        recall = recall_score(y_test, predictions, average="binary", zero_division=0)

        # Enregistrement des métriques dans MLflow
        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("pipeline_path", pipeline_path)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"Metrics logged: Accuracy={accuracy}, Precision={precision}, Recall={recall}")

    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle : {e}")


if __name__ == "__main__":
    # Exemple de données
    sample_data = pd.DataFrame({
        "Pregnancies": [6, 1, 8, 2],
        "Glucose": [148, 85, 183, 89],
        "BloodPressure": [72, 66, 64, 66],
        "SkinThickness": [35, 29, 0, 23],
        "Insulin": [0, 0, 0, 94],
        "BMI": [33.6, 26.6, 23.3, 28.1],
        "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167],
        "Age": [50, 31, 32, 21],
        "Outcome": [1, 0, 1, 0]
    })

    pipeline_path = "../models/pipeline.pkl"
    log_experiment(sample_data, target="Outcome", pipeline_path=pipeline_path)
