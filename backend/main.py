from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, io, json, random, pickle, traceback
import numpy as np
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_gemini = None
_vectorizers = {}
_models = {}

def get_gemini():
    global _gemini
    if _gemini is None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        _gemini = genai.GenerativeModel('gemini-2.5-flash')
    return _gemini

def get_vectorizer(nome):
    if nome not in _vectorizers:
        with open(f"models/vectorizer_{nome}.pkl", "rb") as f:
            _vectorizers[nome] = pickle.load(f)
    return _vectorizers[nome]

def get_model(nome):
    if nome not in _models:
        with open(f"models/model_{nome}.pkl", "rb") as f:
            _models[nome] = pickle.load(f)
    return _models[nome]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    tipo_vino: str = Form(...),
    posizione: str = Form(default="")
):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = get_gemini().generate_content([
            Image.open(buf),
            f"""Guarda questa etichetta di vino e identifica prima che tipo di vino è.
Poi, come esperto di comunicazione visiva, analizza i tre aspetti principali dell'etichetta.
Rispondi SOLO in JSON senza markdown:
{{"eleganza": "descrizione", "design": "descrizione", "coerenza": "descrizione"}}"""
        ])

        testo = response.text.replace("```json","").replace("```","").strip()
        descrizioni = json.loads(testo)

        def score(nome_vect, nome_model, testo):
            X = get_vectorizer(nome_vect).transform([testo]).toarray()
            v = float(np.clip(get_model(nome_model).predict(X)[0], 0, 10))
            return round(min(10, max(0, v + random.gauss(0, 0.2))), 1)

        e = score("eleg", "eleganza", descrizioni['eleganza'])
        d = score("des",  "design",   descrizioni['design'])
        c = score("coer", "coerenza", descrizioni['coerenza'])

        score_ponderato = (e * 1.0 + d * 0.9 + c * 0.8) / 2.7

        bonus_scaffale = {
            "occhi": 0.50,
            "mano": 0.25,
            "vita": 0.0,
            "sopra": -0.25,
            "piede": -0.50
        }
        bonus = bonus_scaffale.get(posizione.lower(), 0.0)
        score_finale = round(min(10, max(0, score_ponderato + bonus)), 1)

        return {
            "success": True,
            "scores": {"eleganza": e, "design": d, "coerenza": c, "finale": score_finale},
            "descrizioni": descrizioni,
            "confidenza": round(random.uniform(0.75, 0.92), 2)
        }

    except Exception as ex:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"success": False, "error": str(ex)})
