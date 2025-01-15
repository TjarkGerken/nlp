import os
import sys
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
data_dir = os.path.abspath(os.path.join(current_dir, "../data"))
sys.path.append(data_dir)
from data.preprocessing import NLTKTokenizer
tokenizer = NLTKTokenizer()

model_dir = os.path.join(parent_dir, "models")
# with open(os.path.join(model_dir, 'logistic_regression.pkl'), 'rb') as file:
#     loaded_model = pickle.load(file)

with open(os.path.join(model_dir, 'tfidf.pkl'), 'rb') as file:
    loaded_tfidf = pickle.load(file)

logger = logging.getLogger('uvicorn')
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def get_status():
    logger.info("GET / endpoint accessed")
    return True

@app.post("/")
def predict_text(req: TextRequest, model = "logR"):
    try:        
        cleaned_text = tokenizer.transform(req.text)
        data = loaded_tfidf.transform([cleaned_text])
        
        loaded_model = load_model(model)
        predictions = loaded_model.predict(data)
        probabilities = loaded_model.predict_proba(data)

        if len(predictions) > 0:
            result = {
                "prediction": "REAL" if predictions[0] == 1 else "FAKE",
                "probability": {
                    "FAKE": probabilities[0][0],
                    "REAL": probabilities[0][1]
                }
            }
            return result
        else:
            return {"error": "No prediction made"}
    except Exception as ex:
        raise HTTPException(status_code=409, detail=ex)
    
def load_model(model = "logR"):
    path = ""
    norm_model = model.lower()
    if norm_model == "logr":
        path = "logistic_regression.pkl"
    elif norm_model == "mlp":
        path = "mlp.pkl"
    elif norm_model == "rnn":
        path = "rnn.pkl"

    with open(os.path.join(model_dir, path), 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

    
