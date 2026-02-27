# 🛡 VERIFACT — AI-Powered Intelligence Verification System

> **For educational purposes only. Always verify claims from primary sources.**

VERIFACT is a full-stack fake-news detection application that combines a fine-tuned **RoBERTa BERT model**, **multi-source evidence retrieval**, and **LLM-generated explanations** behind a sleek cyberpunk Streamlit interface.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Scoring Logic](#scoring-logic)
- [Source Credibility Weights](#source-credibility-weights)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Limitations & Disclaimer](#limitations--disclaimer)

---

## Features

| Feature | Description |
|---|---|
| 🤖 **BERT Classification** | RoBERTa model (`hamzab/roberta-fake-news-classification`) scores text as REAL or FAKE |
| 🔍 **Claim Extraction** | LLM (OpenRouter / Gemini) parses 3–5 verifiable claims from input text |
| 📰 **Multi-Source Evidence** | Searches Wikipedia, NewsAPI, and Google Custom Search |
| ⚖️ **Weighted Decision Engine** | Combines BERT (60%) + Evidence (40%) into a final verdict |
| 💬 **AI Explanation** | OpenRouter / Gemini generates a plain-English verdict summary |
| 📊 **Rich Visualizations** | Gauge chart, score breakdown, source pie chart, credibility bar chart |
| 🌐 **REST API** | FastAPI backend with `/analyze` and `/health` endpoints |
| 🎨 **Cyberpunk UI** | Dark-mode Streamlit frontend with custom CSS |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND (app.py)              │
│    User Input → API Call → Result Rendering + Charts        │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP POST /analyze
┌───────────────────────────▼─────────────────────────────────┐
│                   FASTAPI BACKEND (main.py)                  │
│                                                             │
│  1. BERTEngine          RoBERTa fake-news classifier        │
│     └── local → HF API → Gemini → heuristic fallback       │
│                                                             │
│  2. ClaimExtractor      LLM-powered claim parsing           │
│     └── OpenRouter → Gemini → regex/NLP fallback           │
│                                                             │
│  3. EvidenceSearcher    Multi-source retrieval (async)      │
│     └── Wikipedia → NewsAPI → Google CSE → demo fallback   │
│                                                             │
│  4. EvidenceScorer      Semantic similarity scoring         │
│     └── sentence-transformers → keyword overlap fallback   │
│                                                             │
│  5. DecisionEngine      Weighted verdict + explanation      │
│     └── final = BERT×0.60 + Evidence×0.40                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

**Backend**
- Python 3.10+
- [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- [Transformers](https://huggingface.co/docs/transformers) (RoBERTa)
- [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2)
- [google-generativeai](https://ai.google.dev/) (Gemini 2.5 Flash)
- Pydantic v2

**Frontend**
- [Streamlit](https://streamlit.io/)
- Matplotlib (charts)
- Requests

**External APIs** *(all optional — system degrades gracefully)*
- HuggingFace Inference API
- OpenRouter (free LLM tier: LLaMA 3.1, Mistral 7B, Gemma 3)
- Google Gemini 2.5 Flash
- NewsAPI
- Google Custom Search

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/verifact.git
cd verifact

pip install fastapi uvicorn transformers torch sentence-transformers \
            google-generativeai pydantic python-dotenv requests \
            streamlit matplotlib numpy wikipedia-api
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)
```

### 3. Start the Backend

```bash
python main.py
# Wait for: ✅ All components ready — VERIFACT is live
```

### 4. Start the Frontend (in a separate terminal)

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Backend host/port
API_HOST=localhost
API_PORT=8001

# BERT classification (pick at least one)
HF_API_KEY=hf_xxxx              # HuggingFace Inference API key

# LLM for claim extraction & explanations (pick at least one)
OPENROUTER_API_KEY=sk-or-xxxx   # openrouter.ai (free tier available)
GEMINI_API_KEY=AIzaxxxx          # Google AI Studio (free tier available)

# Evidence search (both optional — falls back to demo links)
NEWS_API_KEY=xxxx                # newsapi.org
GOOGLE_API_KEY=AIzaxxxx          # Google Cloud API key
GOOGLE_CSE_ID=xxxx               # Google Custom Search Engine ID

# Debug (optional)
DEBUG=false
```

**Minimum viable setup:** No API keys required. The system will use local BERT (if transformers is installed), regex-based claim extraction, Wikipedia search, and keyword-based scoring.

---

## API Reference

### `GET /health`

Returns component status.

```json
{
  "status": "healthy",
  "components": {
    "bert": true,
    "bert_method": "local",
    "claim_extractor": true,
    "evidence_searcher": true,
    "evidence_scorer": true,
    "decision_engine": true
  }
}
```

### `POST /analyze`

**Request:**
```json
{
  "text": "NASA confirmed evidence of liquid water on Mars subsurface."
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "label": "REAL",
  "final_score": 0.712,
  "bert_score": 0.810,
  "evidence_score": 0.560,
  "confidence_percent": 71,
  "explanation": "The ML model classified this as likely real with 81% confidence...",
  "claims_extracted": ["NASA confirmed liquid water on Mars subsurface"],
  "evidence_summary": [...],
  "claims_checked": 1,
  "articles_found": 4,
  "processing_time_ms": 1823,
  "bert_method": "local"
}
```

**Validation:**
- `text` must be 3–10,000 characters
- Returns HTTP 400 for invalid input, 500 for processing errors

---

## Scoring Logic

```
final_score = (BERT_score × 0.60) + (evidence_score × 0.40)

label:
  REAL      → final_score ≥ 0.65
  UNCERTAIN → 0.40 < final_score < 0.65
  FAKE      → final_score ≤ 0.40
```

- **BERT score** is the model's probability that the text is REAL
- **Evidence score** is the weighted semantic similarity of the top-3 retrieved articles against each extracted claim, averaged across all claims

---

## Source Credibility Weights

The evidence score is weighted by source credibility:

| Source | Weight |
|---|---|
| Reuters, AP News | 1.00 |
| PolitiFact, Snopes, FactCheck.org | 0.95 |
| BBC | 0.97 |
| The Guardian, NYT, WaPo, NPR | 0.90–0.92 |
| Wikipedia | 0.75 |
| CNN | 0.75 |
| Fox News | 0.68 |
| Breitbart | 0.28 |
| Unknown/other | 0.50 |

---

## Project Structure

```
verifact/
├── main.py          # FastAPI backend — all ML engines
│   ├── BERTEngine           # RoBERTa classification
│   ├── ClaimExtractor       # LLM-based claim parsing
│   ├── EvidenceSearcher     # Wikipedia / NewsAPI / Google
│   ├── EvidenceScorer       # Semantic similarity ranking
│   └── DecisionEngine       # Weighted verdict + explanation
├── app.py           # Streamlit frontend — UI + charts
├── .env             # API keys (not committed)
├── .env.example     # Template
└── README.md
```

---

## How It Works

1. **User submits text** (article, headline, or claim) via the Streamlit UI.
2. **BERT Engine** classifies the raw text using RoBERTa fine-tuned on fake news. Falls back to HF API → Gemini → keyword heuristic if local model unavailable.
3. **Claim Extractor** uses an LLM to pull out 3–5 specific, verifiable factual claims from the text.
4. **Evidence Searcher** queries Wikipedia (always), NewsAPI, and Google CSE concurrently for each extracted claim.
5. **Evidence Scorer** ranks each article against its claim using cosine similarity of sentence-transformer embeddings (or keyword overlap fallback), weighted by source credibility.
6. **Decision Engine** computes `final = BERT×0.6 + Evidence×0.4`, applies thresholds, and generates an LLM explanation.
7. **Frontend** renders the verdict, confidence gauge, score breakdown chart, source pie chart, credibility comparison chart, claim list, and source list.

---

## Limitations & Disclaimer

- **Educational purposes only.** Do not use VERIFACT as the sole arbiter of truth.
- BERT model accuracy varies by text type and domain.
- Evidence search quality depends on API key availability.
- LLM explanations can hallucinate details — treat them as summaries, not ground truth.
- Source credibility weights are opinionated defaults; adjust them in `SOURCE_WEIGHTS` to fit your use case.
- The system does not retain logs or store submitted text.
