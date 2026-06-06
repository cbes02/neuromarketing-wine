# NeuroLabel — Wine Label Intelligence

Applicazione web che analizza le etichette dei vini tramite intelligenza artificiale e restituisce un punteggio visivo basato sulla metodologia neuromappa IULM.

---

## App live

🔗 **https://cbes02.github.io/neuromarketing-wine**

L'app è già deployata e funzionante — basta aprire il link da qualsiasi dispositivo, nessuna installazione richiesta.

---

## Come funziona

1. L'utente carica o scatta una foto dell'etichetta fronte
2. Il backend chiama **Gemini 2.5 Flash** con un prompt calibrato sui criteri IULM
3. Le descrizioni vengono vettorizzate con **TF-IDF**
4. Tre modelli **Gradient Boosting** predicono i punteggi di eleganza, design e coerenza cromatica
5. Il punteggio finale viene calcolato tramite **media ponderata** con pesi neurologici + bonus/malus posizione scaffale

---

## Struttura del repo

```
neuromarketing-wine/
├── index.html              # Frontend (GitHub Pages)
└── backend/
    ├── main.py             # API FastAPI
    ├── requirements.txt
    └── models/
        ├── model_eleganza.pkl
        ├── model_design.pkl
        ├── model_coerenza.pkl
        └── tfidf_vectorizer.pkl
```

---

## Esecuzione in locale

Se si vuole eseguire il backend localmente invece di usare quello già deployato su Render:

**1. Clona il repo**
```bash
git clone https://github.com/cbes02/neuromarketing-wine.git
cd neuromarketing-wine/backend
```

**2. Installa le dipendenze**
```bash
pip install -r requirements.txt
```

**3. Crea un file `.env` nella cartella `backend/` con la tua chiave Gemini**
```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```
La chiave si ottiene gratuitamente su [Google AI Studio](https://aistudio.google.com).

**4. Avvia il backend**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**5. Aggiorna il frontend**
Nel file `index.html` sostituisci la riga:
```javascript
const BACKEND = 'https://neuromarketing-wine.onrender.com';
```
con:
```javascript
const BACKEND = 'http://localhost:8000';
```
Poi apri `index.html` nel browser.

---

## API Keys necessarie

- **Gemini API Key** → [Google AI Studio](https://aistudio.google.com) — gratuita, 1500 chiamate/giorno

---

## Tecnologie

- **Frontend:** HTML/CSS/JavaScript — GitHub Pages
- **Backend:** FastAPI (Python) — Render.com
- **AI:** Gemini 2.5 Flash (Google)
- **ML:** Gradient Boosting, TF-IDF, SMOTE, Sentence Transformers
- **Training:** Google Colab GPU T4

---

## Progetto

NeuroMarketing — IULM Milano — Chiara Beretta & Rebecca Magarotto 2026
