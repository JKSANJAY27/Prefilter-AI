# Prefilter AI Platform

The **Prefilter AI Platform** is the server layer that exposes the full query understanding middleware pipeline via a REST API and a real-time web dashboard.

---

## Quick Start

### 1. Install dependencies

```bash
# Core (already installed if you did pip install -e .)
pip install prefilter-ai

# Platform extras (FastAPI for full REST API)
pip install fastapi uvicorn[standard]
```

### 2. Start the server

```bash
# From the project root:
python -m platform.server

# Or with uvicorn directly (auto-reload on code changes):
uvicorn platform.server:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Open the dashboard

```
http://localhost:8080/
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard UI |
| `GET` | `/health` | Health check |
| `POST` | `/v1/parse` | Single query pipeline |
| `POST` | `/v1/session/{id}` | Multi-turn session query |
| `DELETE` | `/v1/session/{id}` | Clear session |
| `GET` | `/v1/domains` | List domain schemas |
| `GET` | `/v1/analytics` | Runtime metrics |
| `GET` | `/docs` | Interactive API docs (FastAPI) |

---

## REST API Usage

### Single query — `POST /v1/parse`

```bash
curl -X POST http://localhost:8080/v1/parse \
  -H "Content-Type: application/json" \
  -d '{"query": "gaming laptop under $800", "parser": "spacy"}'
```

**Response:**
```json
{
  "query": "gaming laptop under $800",
  "domain": "ecommerce",
  "filters": [
    {"field": "product", "operator": "eq", "value": "laptop", "confidence": 0.81},
    {"field": "price", "operator": "lt", "value": 800.0, "confidence": 0.87}
  ],
  "preferences": [
    {"field": "feature", "value": "Dedicated GPU", "weight": 0.9}
  ],
  "conflicts": [
    "Feasibility conflict: Gaming/RTX laptops start at ~$700–$900. Budget $800 is below market floor."
  ],
  "relaxed": {
    "filters": [
      {"field": "price", "operator": "lt", "value": 1120.0}
    ]
  },
  "sql": "price < 800.0",
  "elasticsearch": {"bool": {"filter": [{"range": {"price": {"lt": 800.0}}}]}},
  "mongodb": {"price": {"$lt": 800.0}},
  "explanation": {
    "price": "'price' must be less than 800.0 — extracted via spaCy extractor (confidence: 87%)"
  },
  "total_latency_ms": 4.2
}
```

### Multi-turn session — `POST /v1/session/{id}`

```bash
SESSION_ID="my-session-001"

# Turn 1
curl -X POST http://localhost:8080/v1/session/$SESSION_ID \
  -d '{"query": "gaming laptops", "parser": "spacy"}'

# Turn 2 — refines turn 1
curl -X POST http://localhost:8080/v1/session/$SESSION_ID \
  -d '{"query": "Only Lenovo, 32GB RAM"}'

# Turn 3 — adds constraint, re-runs full pipeline
curl -X POST http://localhost:8080/v1/session/$SESSION_ID \
  -d '{"query": "Under $1200"}'

# Clear session
curl -X DELETE http://localhost:8080/v1/session/$SESSION_ID
```

---

## Dashboard Views

| View | What it shows |
|---|---|
| **Pipeline** | Real-time stage visualizer, conflict banners, all DSL translations |
| **Session** | Multi-turn conversational search with turn history |
| **Analytics** | Conflict rate, relaxation rate, avg latency, domain distribution |
| **Domains** | Domain registry with fields, types, and importance weights |

---

## Parser Options

| Value | Description | Requires |
|---|---|---|
| `spacy` | Rule-based NER + regex (~1–5ms) | `pip install spacy && python -m spacy download en_core_web_sm` |
| `slm` | Local fine-tuned Qwen 0.8B LoRA | `pip install torch peft transformers` + adapter files in `hg-face/` |
| `gemini` | Gemini API extraction | `GEMINI_API_KEY` env var |

---

## SLM (Fine-Tuned Model) Setup

The fine-tuned Qwen 0.8B adapter lives in `hg-face/json/`. To use it:

```bash
pip install torch peft transformers
```

Then in your request:
```json
{"query": "...", "parser": "slm"}
```

To make it available to anyone (not just local users), upload to Hugging Face:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='hg-face/json',
    repo_id='JKSANJAY27/prefilter-ai-json-0.8b',
    repo_type='model',
)
"
```

The loader in `prefilter_ai/config.py` already points to that Hub ID — it will load remotely after the upload.

---

## Architecture

```
User Query
    │
    ▼
┌───────────────────────┐
│   PrefilterPipeline   │  ← public API
└───────────────────────┘
    │
    ├─ 1. Parser (spaCy / SLM / Gemini)
    │       NL → IRFilterConstraints
    │
    ├─ 2. OntologyEngine
    │       implicit prefs inference (40+ rules, 10 domains)
    │
    ├─ 3. ConflictDetector
    │       feasibility checks, contradiction detection
    │
    ├─ 4. QueryRelaxer
    │       conflict-targeted constraint relaxation
    │
    ├─ 5. Backend Translators (lazy)
    │       IR → SQL / Elasticsearch / MongoDB / ChromaDB
    │
    └─ 6. Explainer
            per-field provenance dict

FastAPI Platform Server
    │
    ├─ POST /v1/parse
    ├─ POST /v1/session/{id}  (stateful, multi-turn)
    ├─ GET  /v1/analytics
    └─ GET  /                 (dashboard)
```
