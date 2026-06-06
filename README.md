# NeuroLabel — Wine Label Intelligence

Applicazione web che analizza le etichette dei vini tramite intelligenza artificiale e restituisce un punteggio visivo basato sulla metodologia neuromappa IULM.

🔗 **App live:** https://cbes02.github.io/neuromarketing-wine

---

## Come funziona

1. L'utente carica o scatta una foto dell'etichetta
2. Il backend chiama **Gemini 2.5 Flash** con un prompt calibrato sui criteri IULM
3. Le descrizioni vengono vettorizzate con **TF-IDF**
4. Tre modelli **Gradient Boosting** predicono i punteggi di eleganza, design e coerenza
5. Il punteggio finale viene calcolato tramite **media ponderata** con pesi neurologici + bonus/malus posizione scaffale

---

## Struttura del repo

```
neuromarketing-wine/
├── index.html
└── backend/
    ├── main.py
    ├── requirements.txt
    └── models/
        ├── model_eleganza.pkl
        ├── model_design.pkl
        ├── model_coerenza.pkl
        └── tfidf_vectorizer.pkl
```

---

## Esecuzione locale

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY="la_tua_chiave"
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Keys necessarie

- **Gemini API Key** → [Google AI Studio](https://aistudio.google.com) — variabile d'ambiente `GEMINI_API_KEY`

---

## Tecnologie

- **Frontend:** HTML/CSS/JavaScript — GitHub Pages
- **Backend:** FastAPI (Python) — Render.com
- **AI:** Gemini 2.5 Flash (Google)
- **ML:** Gradient Boosting, TF-IDF, SMOTE, Sentence Transformers
- **Training:** Google Colab GPU T4

---

## Progetto

NeuroMarketing — IULM Milano — Chiara Beretta 2026
