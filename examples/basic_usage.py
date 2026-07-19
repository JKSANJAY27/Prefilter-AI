"""
examples/basic_usage.py
-----------------------
Demonstrates the core prefilter-ai API without requiring a GPU.
The model-loading lines are commented out so this file runs in any
environment; uncomment them when you have a GPU available.
"""

# ── Imports ───────────────────────────────────────────────────
from prefilter_ai import PrefilterAI, ModelFormat, ParseResult

# ── 1. Basic usage (JSON model, default) ──────────────────────
expert = PrefilterAI()
# Under the hood, this will check if the local 'hg-face/json' folder is available
# and load from it directly.
result = expert.parse("noise cancelling headphones under $200")
print("Parsed fields:", result.fields)
# # → {'domain': 'ecommerce', 'product': 'headphones',
# #    'feature': 'noise cancelling', 'price': 'lt:200'}

# ── 2. YAML model ─────────────────────────────────────────────
expert_yaml = PrefilterAI(fmt=ModelFormat.YAML)
# Under the hood, this will check if the local 'hg-face/yaml' folder is available.
result_yaml = expert_yaml.parse("3BR house in Austin under $600k with pool")
print("YAML output:\n", result_yaml.to_yaml())

# ── 3. Numeric constraint decoding ────────────────────────────
result = expert.parse("remote ML engineer job paying over $150k")
salary = result.get_numeric_constraint("salary")
print("Salary constraint:", salary)
# # → {'operator': 'gt', 'value': 150000.0, 'value_hi': None}
#
# # You can use this in a search filter directly:
if salary and salary["operator"] == "gt":
    filter_salary_gt = salary["value"]  # 150000.0

# ── 4. Serialise result ───────────────────────────────────────
print("JSON output:\n", result.to_json(indent=2))
print("YAML output:\n", result.to_yaml())
print("Dict output:\n", result.to_dict())

# ── 5. Query Translators (New features!) ──────────────────────
sql_query, sql_params = result.to_sql("jobs_table")
print("SQL:", sql_query)
print("SQL Params:", sql_params)

mongo_filter = result.to_mongodb()
print("MongoDB Filter:", mongo_filter)

chroma_where = result.to_chromadb()
print("ChromaDB Where Clause:", chroma_where)

# ── 6. Batch parsing ──────────────────────────────────────────
queries = [
    "Python ML course for beginners under $30",
    "5-star hotel in Paris with breakfast under $400/night",
    "Taylor Swift concert in London in July",
]
results = expert.parse_batch(queries)
for r in results:
    print(r.query, "→", r.fields)

# ── 7. Custom adapter ─────────────────────────────────────────
expert = PrefilterAI(model_id="your-org/your-fine-tuned-adapter")

# ── 8. Custom generation config ───────────────────────────────
expert = PrefilterAI(generation_config={"temperature": 0.0, "max_new_tokens": 128})

print("See comments in this file for usage examples.")
print("Uncomment the lines after loading a model on a GPU instance.")
