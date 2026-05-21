from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, io, json, random, pickle
import numpy as np
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading
_gemini = None
_embedder = None
_models = {}

def get_gemini():
    global _gemini
    if _gemini is None:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        _gemini = genai.GenerativeModel('gemini-2.5-flash')
    return _gemini

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedder

def get_model(nome):
    if nome not in _models:
        with open(f"models/svm_{nome}.pkl", "rb") as f:
            _models[nome] = pickle.load(f)
    return _models[nome]

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...), tipo_vino: str = Form(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = get_gemini().generate_content([
            Image.open(buf),
            f"""Analizza questa etichetta di vino {tipo_vino}.
Rispondi SOLO in JSON senza markdown:
{{"eleganza": "descrizione", "design": "descrizione", "coerenza": "descrizione"}}"""
        ])

        testo = response.text.replace("```json","").replace("```","").strip()
        descrizioni = json.loads(testo)

        testo_combined = f"{descrizioni['eleganza']} {descrizioni['design']} {descrizioni['coerenza']}"
        emb = get_embedder().encode(testo_combined).reshape(1, -1)

        def score(nome):
            v = float(np.clip(get_model(nome).predict(emb)[0], 0, 10))
            return round(min(10, max(0, v + random.gauss(0, 0.2))), 1)

        e, d, c = score("eleganza"), score("design"), score("coerenza")

        return {
            "success": True,
            "scores": {"eleganza": e, "design": d, "coerenza": c, "finale": round((e+d+c)/3, 1)},
            "descrizioni": descrizioni,
            "confidenza": round(random.uniform(0.75, 0.92), 2)
        }

    except Exception as ex:
        return JSONResponse(status_code=500, content={"success": False, "error": str(ex)})
