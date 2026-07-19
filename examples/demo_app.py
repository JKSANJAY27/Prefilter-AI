import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import sys

# Add root directory to path to allow import of prefilter_ai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from prefilter_ai import PrefilterAI, ModelFormat
    from prefilter_ai.ontology import OntologyEngine
    from prefilter_ai.validator import ConflictDetector
    from prefilter_ai.relaxer import QueryRelaxer
except Exception as e:
    print(f"Warning: Could not import PrefilterAI: {e}")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prefilter AI — Query Understanding Layer</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #080710;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.15);
            --warning: #ef4444;
            --warning-glow: rgba(239, 68, 68, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.06) 0%, transparent 45%);
        }

        header {
            width: 100%;
            max-width: 1200px;
            padding: 2.5rem 2rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logo-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--accent);
        }

        .badge {
            background: var(--border-color);
            border: 1px solid var(--border-color);
            padding: 0.35rem 0.75rem;
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        main {
            width: 100%;
            max-width: 1200px;
            padding: 0 2rem 4rem;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        @media (min-width: 900px) {
            main {
                grid-template-columns: 1.15fr 0.85fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        textarea {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            color: var(--text-color);
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            height: 80px;
            outline: none;
            transition: all 0.2s ease;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .settings-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
        }

        select {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.75rem;
            color: var(--text-color);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            cursor: pointer;
        }

        select:focus {
            border-color: var(--primary);
        }

        button {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.85rem;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        button:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .examples {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .example-btn {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.65rem 1rem;
            text-align: left;
            font-size: 0.85rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: none;
            font-weight: 400;
        }

        .example-btn:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-color);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .output-section {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .tab-buttons {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            gap: 0.75rem;
            overflow-x: auto;
            padding-bottom: 0.25rem;
        }

        .tab-btn {
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 0.5rem 0.25rem 0.75rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            box-shadow: none;
            border-radius: 0;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: var(--text-color);
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
            box-shadow: none;
        }

        pre {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            min-height: 280px;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .keyword { color: #f43f5e; }
        .string { color: #10b981; }
        .number { color: #f59e0b; }
        .operator { color: #3b82f6; }
        .comment { color: #6b7280; }

        .conflict-box {
            display: none;
            background: var(--warning-glow);
            border: 1px solid var(--warning);
            border-radius: 12px;
            padding: 1rem;
            color: var(--text-color);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        .conflict-title {
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }

        .loader-container {
            display: none;
            align-items: center;
            justify-content: center;
            height: 250px;
            border: 1px solid var(--border-color);
            background: rgba(0, 0, 0, 0.35);
            border-radius: 12px;
        }

        .spinner {
            width: 32px;
            height: 32px;
            border: 3px solid rgba(99, 102, 241, 0.1);
            border-radius: 50%;
            border-top-color: var(--primary);
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <div class="logo-dot"></div>
            Prefilter AI
        </div>
        <div class="badge">Query Understanding Middleware</div>
    </header>

    <main>
        <!-- Left: Controls -->
        <div class="card">
            <div class="card-title">
                <span>Pipeline Console</span>
            </div>
            
            <div class="input-group">
                <label for="query">Search Query</label>
                <textarea id="query" placeholder="Type a natural language query... e.g., gaming laptop under $500"></textarea>
            </div>

            <div class="settings-grid">
                <div class="input-group">
                    <label for="backend">Extraction</label>
                    <select id="backend">
                        <option value="spacy">spaCy Rule-based</option>
                        <option value="slm">SLM Local Model</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="format">Format</label>
                    <select id="format">
                        <option value="json">JSON Mode</option>
                        <option value="yaml">YAML Mode</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="relaxation">Auto Relaxation</label>
                    <select id="relaxation">
                        <option value="0">None</option>
                        <option value="1">Level 1 (Drop Low)</option>
                        <option value="2">Level 2 (+Expand Price)</option>
                        <option value="3">Level 3 (+Drop Med)</option>
                    </select>
                </div>
            </div>

            <button id="parse-btn">Process Pipeline</button>

            <div class="examples">
                <label>Test Pipeline Scenarios</label>
                <button class="example-btn" data-query="gaming laptop under $500">❌ Conflict: gaming laptop under $500</button>
                <button class="example-btn" data-query="laptop for AI and coding">🧠 Ontology: laptop for AI and coding</button>
                <button class="example-btn" data-query="5-star beachfront hotel under $100">🏖️ Conflict + Heuristic: Beachfront hotel under $100</button>
                <button class="example-btn" data-query="flights from JFK to Tokyo under $3,000 in business class">✈️ Multiple Constraints: flight JFK to Tokyo</button>
            </div>
        </div>

        <!-- Right: Output -->
        <div class="card output-section">
            <div class="conflict-box" id="conflict-box">
                <div class="conflict-title">⚠️ Feasibility Conflict Detected</div>
                <div id="conflict-text"></div>
            </div>

            <div class="tab-buttons">
                <button class="tab-btn active" data-tab="ir">Intermediate Rep (IR)</button>
                <button class="tab-btn" data-tab="sql">SQL DSL</button>
                <button class="tab-btn" data-tab="mongodb">MongoDB Filter</button>
                <button class="tab-btn" data-tab="elasticsearch">Elasticsearch DSL</button>
                <button class="tab-btn" data-tab="chromadb">ChromaDB Query</button>
            </div>

            <div id="output-loader" class="loader-container">
                <div class="spinner"></div>
            </div>
            
            <pre id="output-block"></pre>
        </div>
    </main>

    <script>
        const queryInput = document.getElementById('query');
        const backendSelect = document.getElementById('backend');
        const formatSelect = document.getElementById('format');
        const relaxationSelect = document.getElementById('relaxation');
        const parseBtn = document.getElementById('parse-btn');
        const outputBlock = document.getElementById('output-block');
        const loader = document.getElementById('output-loader');
        const tabBtns = document.querySelectorAll('.tab-btn');
        const exampleBtns = document.querySelectorAll('.example-btn');
        const conflictBox = document.getElementById('conflict-box');
        const conflictText = document.getElementById('conflict-text');

        let activeTab = 'ir';
        let latestData = null;

        // Examples
        exampleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                queryInput.value = btn.getAttribute('data-query');
                parseQuery();
            });
        });

        // Tabs
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                activeTab = btn.getAttribute('data-tab');
                renderOutput();
            });
        });

        parseBtn.addEventListener('click', parseQuery);

        async function parseQuery() {
            const query = queryInput.value.trim();
            if (!query) return;

            outputBlock.style.display = 'none';
            conflictBox.style.display = 'none';
            loader.style.display = 'flex';

            try {
                const response = await fetch(`/parse?query=${encodeURIComponent(query)}&backend=${backendSelect.value}&format=${formatSelect.value}&relax=${relaxationSelect.value}`);
                latestData = await response.json();
                
                // Show conflicts if present
                if (latestData.ir && latestData.ir.conflicts && latestData.ir.conflicts.length > 0) {
                    conflictBox.style.display = 'block';
                    conflictText.innerHTML = latestData.ir.warnings.join('<br>');
                }

                renderOutput();
            } catch (err) {
                outputBlock.innerHTML = `<span class="keyword">Error:</span> Could not communicate with local backend.`;
            } finally {
                loader.style.display = 'none';
                outputBlock.style.display = 'block';
            }
        }

        function renderOutput() {
            if (!latestData) {
                outputBlock.innerHTML = '// Click "Process Pipeline" or pick an example to begin.';
                return;
            }

            if (latestData.error) {
                outputBlock.innerHTML = `<span class="keyword">Error:</span> ${latestData.error}`;
                return;
            }

            if (activeTab === 'ir') {
                outputBlock.innerHTML = highlightSyntax(JSON.stringify(latestData.ir, null, 2));
            } else if (activeTab === 'sql') {
                const sqlText = `-- Generated parameterized SQL Query\\n\\n${latestData.sql.query}\\n\\n-- Parameters:\\n${JSON.stringify(latestData.sql.params, null, 2)}`;
                outputBlock.innerHTML = highlightSyntax(sqlText);
            } else if (activeTab === 'mongodb') {
                outputBlock.innerHTML = highlightSyntax(JSON.stringify(latestData.mongodb, null, 2));
            } else if (activeTab === 'chromadb') {
                outputBlock.innerHTML = highlightSyntax(JSON.stringify(latestData.chromadb, null, 2));
            } else if (activeTab === 'elasticsearch') {
                outputBlock.innerHTML = highlightSyntax(JSON.stringify(latestData.elasticsearch, null, 2));
            }
        }

        function highlightSyntax(jsonStr) {
            return jsonStr
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/("(\\u[a-zA-Z0-9]{4}|\\\\[^u]|[^\\\\"])*"(\\s*:)?|\\b(true|false|null)\\b|-?\\d+(?:\\.\\d*)?(?:[eE][+-]?\\d+)?)/g, function (match) {
                    let cls = 'number';
                    if (/^"/.test(match)) {
                        if (/:$/.test(match)) {
                            cls = 'keyword';
                        } else {
                            cls = 'string';
                        }
                    } else if (/true|false/.test(match)) {
                        cls = 'operator';
                    } else if (/null/.test(match)) {
                        cls = 'comment';
                    }
                    return '<span class="' + cls + '">' + match + '</span>';
                });
        }

        // Initialize empty
        renderOutput();
    </script>
</body>
</html>
"""

class DemoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif path == "/parse":
            query_params = urllib.parse.parse_qs(url_parsed.query)
            query = query_params.get("query", [""])[0]
            backend = query_params.get("backend", ["spacy"])[0]
            fmt_str = query_params.get("format", ["json"])[0]
            relax_val = int(query_params.get("relax", ["0"])[0])

            if not query:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty query"}).encode("utf-8"))
                return

            try:
                # 1. Extraction via dynamically configured parser
                parser = PrefilterAI(fmt=fmt_str, parse_backend=backend)
                result = parser.parse(query)
                ir = result._get_or_create_ir()

                # 2. Ontology Soft Preference Inference
                ir = OntologyEngine().infer(ir, query)

                # 3. Feasibility Contradiction Checking
                ir = ConflictDetector().validate(ir)

                # 4. Optional Relaxation
                if relax_val > 0:
                    ir = QueryRelaxer().relax(ir, relaxation_level=relax_val)

                # 5. Translations
                from prefilter_ai.translators.sql import SQLTranslator
                from prefilter_ai.translators.mongodb import MongoDBTranslator
                from prefilter_ai.translators.chromadb import ChromaDBTranslator
                from prefilter_ai.translators.elasticsearch import ElasticsearchTranslator

                sql_q, sql_params = SQLTranslator().translate(ir)

                response_data = {
                    "ir": ir.to_dict(),
                    "sql": {
                        "query": sql_q,
                        "params": sql_params
                    },
                    "mongodb": MongoDBTranslator().translate(ir),
                    "chromadb": ChromaDBTranslator().translate(ir),
                    "elasticsearch": ElasticsearchTranslator().translate(ir)
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server = HTTPServer(("localhost", port), DemoRequestHandler)
    print(f"🚀 Demo server started at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
