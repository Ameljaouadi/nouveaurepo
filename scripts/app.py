from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configuration du dossier de templates
templates = Jinja2Templates(directory="templates")
# Chemin local vers le meilleur modèle sauvegardé
best_model_path = "../mymodel/best_model.pkl"

mlflow.set_tracking_uri("http://localhost:5000")

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dict_res = {0: 'Not-Diabetes', 1: 'Diabetes'}

#pipeline_path = '../models/pipeline.pkl'
#with open(pipeline_path, 'rb') as pipeline_file:
    #pipeline = pickle.load(pipeline_file)
# Fonction pour charger le meilleur modèle depuis MLflow
# Fonction pour charger le modèle depuis un fichier local
def load_local_model(model_path: str):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"Modèle chargé avec succès depuis : {model_path}")
        return model
    except Exception as e:
        print("Erreur lors du chargement du modèle :", str(e))
        raise Exception("Impossible de charger le modèle localement.")

# Charger le modèle au démarrage de l'application
try:
    pipeline = load_local_model(best_model_path)
except Exception as e:
    print(e)
    pipeline = None

class DataInput(BaseModel):
    data: list

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: DataInput):
    try:
        print(f"Received data: {input_data}")
        df = pd.DataFrame(input_data.data, columns=columns)
        predictions = pipeline.predict(df)
        results = [dict_res[pred] for pred in predictions]
        return {"predictions": results}
    
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
