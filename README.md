<div align="center">

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/prefilter--ai-0.1.2-0f0f0f?style=for-the-badge&labelColor=0f0f0f&color=39d353&logoColor=white">
  <img alt="prefilter-ai" src="https://img.shields.io/badge/prefilter--ai-0.1.2-fafafa?style=for-the-badge&labelColor=fafafa&color=16a34a&logoColor=black">
</picture>

<br/><br/>

<h1>
  <img src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Magnifying%20glass%20tilted%20right/3D/magnifying_glass_tilted_right_3d.png" width="36" align="center" />
  &nbsp;prefilter-ai
</h1>

<p><strong>Natural language → structured search queries, instantly.</strong><br/>
Fine-tuned Qwen3.5-0.8B LoRA adapters for search query parsing across 10 domains.</p>

<br/>

[![GitHub stars](https://img.shields.io/github/stars/JKSANJAY27/Prefilter-AI?style=flat-square&color=16a34a)](https://github.com/JKSANJAY27/Prefilter-AI/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square&labelColor=fefce8)](LICENSE)
[![Open In Colab](https://img.shields.io/badge/Colab-Try_it_now-F9AB00?style=flat-square&logo=googlecolab&logoColor=white&labelColor=2d2d2d)](https://colab.research.google.com/github/JKSANJAY27/Prefilter-AI/blob/main/examples/prefilter_ai_colab.ipynb)

<br/>

</div>

---

<br/>

## What it does

```
"Non-stop business class from JFK to Tokyo under $3,000"
```

```json
{
  "domain":      "flights",
  "origin":      "JFK",
  "destination": "Tokyo",
  "cabin_class": "business",
  "stops":       "lte:0",
  "price":       "lt:3000"
}
```

Prefilter AI uses a custom fine-tuned Small Language Model to understand natural language queries and extract **only the fields explicitly mentioned** — never hallucinating values that aren't there. It works across 10 search verticals out of the box.

<br/>

## Install

```bash
pip install .
```
<br/>

## Usage

### Basic

```python
from prefilter_ai import PrefilterAI, ModelFormat, ParseResult

expert = PrefilterAI()  # loads the JSON adapter by default (checks local hg-face/json folder first)

result = expert.parse("noise cancelling headphones any colour but red or green, under $200")
print(result.fields)
```

```python
{
    'domain':  'ecommerce',
    'product': 'headphones',
    'feature': 'noise cancelling',
    'color':   ['ne:red', 'ne:green'],
    'price':   'lt:200'
}
```

### Built-in Query Translators (SQL, MongoDB, ChromaDB)
Translate natural language queries directly to standard DB filter clauses:

```python
# 1. SQL Where Clauses (parameterized)
sql_query, sql_params = result.to_sql("products")
# "SELECT * FROM products WHERE product = :product AND price < :price ..."

# 2. MongoDB filter dicts
mongo_filter = result.to_mongodb()
# {"product": "headphones", "price": {"$lt": 200.0}, ...}

# 3. ChromaDB metadata queries
chroma_where = result.to_chromadb()
# {"$and": [{"product": {"$eq": "headphones"}}, ...]}
```

<br/>

## Why hybrid search beats the alternatives

| | Text-to-SQL | Pure vector search | Hybrid search (this pipeline) |
|---|---|---|---|
| Hard constraints (price, brand, color) | ✅ | ❌ | ✅ |
| Semantic intent ("good for travel") | ❌ | ✅ | ✅ |
| Ranked results by relevance | ❌ | ✅ | ✅ |
| Works on unstructured descriptions | ❌ | ✅ | ✅ |
| Respects exclusions ("not black") | ✅ | ❌ | ✅ |
| Price is a hard cutoff, not a soft signal | ✅ | ❌ | ✅ |

Text-to-SQL is a **lookup tool** — it returns rows that match, but can't rank by relevance or understand semantics.  
Pure vector search is a **semantic tool** — it understands meaning, but treats "$200" as a soft hint, not a hard rule. A $350 product can rank above a $180 one if its description is more similar to the query.  
This pipeline is a **retrieval tool** — structured filters enforce the hard constraints first, then vector search ranks the surviving candidates by semantic relevance.

In production, we generally use a version of this pattern:  
**structured pre-filtering → ANN (approximate nearest neighbour) vector search → learning-to-rank re-ranker**.

prefilter-ai makes step 1 trivial with a tiny, fast, locally-runnable model.

<br/>

## Interactive Web Demo
Run our interactive playground to test queries, switch backends (SLM vs spaCy), and see translated query filters in real-time:
```bash
python examples/demo_app.py
```
Then visit [http://localhost:8080](http://localhost:8080) in your browser.

<br/>

## Operator reference

All numeric and exclusion constraints use a consistent prefix so downstream filters need zero NLP — just parse the string.

| Query phrase | Output value |
|---|---|
| `"under $200"`, `"below $200"` | `lt:200` |
| `"up to $200"`, `"max $200"` | `lte:200` |
| `"over $150k"`, `"above $150k"` | `gt:150000` |
| `"at least $150k"`, `"$150k+"` | `gte:150000` |
| `"around $200"`, `"~$200"` | `approx:200` |
| `"$100–$200"`, `"between $100 and $200"` | `between:100:200` |
| `"any colour but red or green"` | `["ne:red", "ne:green"]` |

**Applying a filter in one line:**

```python
result = expert.parse("apartments under $2,500/month in Austin")
salary = result.get_numeric_constraint("price")
# {'operator': 'lt', 'value': 2500.0, 'value_hi': None}

filtered = [l for l in listings if l["price"] < salary["value"]]
```

<br/>

## Repo structure

```
prefilter-ai/
├── prefilter_ai/         # Library source
│   ├── __init__.py
│   ├── expert.py         # PrefilterAI class (main API)
│   ├── config.py         # Model IDs, prompts, format enum, local hg-face detection
│   ├── loader.py         # Model loading (unsloth / peft / plain)
│   ├── parser.py         # Raw output → dict parsers (with JSON repair)
│   ├── result.py         # ParseResult dataclass + query translators
│   └── exceptions.py     # Custom exceptions
├── training/             # Fine-tuning pipeline
│   ├── finetune.py       # Training script
│   └── evaluate.py       # Format comparison leaderboard
├── tests/
│   └── test_prefilter_ai.py
├── examples/
│   ├── demo_app.py       # Interactive playground UI
│   ├── basic_usage.py
│   ├── prefilter_ai_colab.ipynb
│   └── ecommerce/        # E-commerce ChromaDB example
├── pyproject.toml
└── README.md
```

<br/>

## Development

```bash
git clone https://github.com/JKSANJAY27/Prefilter-AI
cd prefilter-ai
pip install -e ".[dev]"

pytest tests/ -v                                          # unit tests (no GPU needed)
PREFILTER_AI_RUN_MODEL_TESTS=1 pytest tests/ -v           # includes model inference tests
```

<br/>

## License

MIT © [Sanjay JK](https://github.com/JKSANJAY27)