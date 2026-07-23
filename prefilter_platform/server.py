"""
platform/server.py — Prefilter AI Platform REST API Server.

Serves the middleware pipeline via HTTP, so any client (frontend, microservice,
or mobile app) can query it over a standard REST interface.

Endpoints
---------
POST /v1/parse              — Run single query through full pipeline
POST /v1/session/{id}       — Multi-turn conversational search
DELETE /v1/session/{id}     — Clear a session
GET  /v1/domains            — List registered domains with schemas
GET  /v1/health             — Health check
GET  /                      — Serve the dashboard UI

Run
---
    python -m platform.server
    # or
    uvicorn platform.server:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prefilter.platform")

# ── Try FastAPI, fall back to stdlib http.server ─────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    _FASTAPI = True
except ImportError:
    _FASTAPI = False

from prefilter_ai import PipelineSession, PrefilterPipeline
from prefilter_ai.registry import SchemaRegistry

# ── Shared pipeline & session store ───────────────────────────────────

_PIPELINE_CACHE: dict[str, PrefilterPipeline] = {}
_SESSION_STORE: dict[str, PipelineSession] = {}
_ANALYTICS: dict[str, Any] = {
    "total_queries": 0,
    "domain_counts": defaultdict(int),
    "conflict_count": 0,
    "relaxation_count": 0,
    "latency_samples": [],
    "zero_result_queries": [],
}
_DASHBOARD_DIR = Path(__file__).parent / "dashboard"


def _get_pipeline(parser: str = "spacy") -> PrefilterPipeline:
    if parser not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[parser] = PrefilterPipeline(parser=parser)
    return _PIPELINE_CACHE[parser]


def _record_analytics(result_dict: dict) -> None:
    _ANALYTICS["total_queries"] += 1
    _ANALYTICS["domain_counts"][result_dict.get("domain", "general")] += 1
    if result_dict.get("conflicts"):
        _ANALYTICS["conflict_count"] += 1
    if result_dict.get("relaxed"):
        _ANALYTICS["relaxation_count"] += 1
    total_lat = result_dict.get("total_latency_ms", 0)
    if total_lat:
        _ANALYTICS["latency_samples"].append(total_lat)
        if len(_ANALYTICS["latency_samples"]) > 1000:
            _ANALYTICS["latency_samples"] = _ANALYTICS["latency_samples"][-500:]


# ── FastAPI App ────────────────────────────────────────────────────────

if _FASTAPI:
    app = FastAPI(
        title="Prefilter AI Platform",
        description="AI Query Understanding Middleware — REST API",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request / Response models ──────────────────────────────────────

    class ParseRequest(BaseModel):
        query: str
        parser: str = "spacy"
        relaxation_level: int = 1

    class SessionQueryRequest(BaseModel):
        query: str
        parser: str = "spacy"

    # ── Routes ────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_dashboard():
        index = _DASHBOARD_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>Prefilter AI Platform</h1><p>Dashboard not found. Check platform/dashboard/index.html</p>"
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    @app.post("/v1/parse")
    async def parse_query(req: ParseRequest):
        try:
            pipeline = _get_pipeline(req.parser)
            result = pipeline.run(req.query)
            d = result.to_dict()
            _record_analytics(d)
            return JSONResponse(content=d)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Pipeline error for query: %s", req.query)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/session/{session_id}")
    async def session_query(session_id: str, req: SessionQueryRequest):
        if session_id not in _SESSION_STORE:
            pipeline = _get_pipeline(req.parser)
            _SESSION_STORE[session_id] = pipeline.new_session()

        session = _SESSION_STORE[session_id]
        try:
            result = session.run(req.query)
            d = result.to_dict()
            d["session_id"] = session_id
            d["turn"] = len(session.history)
            _record_analytics(d)
            return JSONResponse(content=d)
        except Exception as e:
            logger.exception("Session error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/v1/session/{session_id}")
    async def delete_session(session_id: str):
        if session_id in _SESSION_STORE:
            _SESSION_STORE[session_id].reset()
            del _SESSION_STORE[session_id]
        return {"status": "cleared", "session_id": session_id}

    @app.get("/v1/domains")
    async def list_domains():
        registry = SchemaRegistry()
        domains = {}
        for name in registry.list_domains():
            schema = registry.get(name)
            domains[name] = {
                "description": schema.description,
                "fields": {
                    fname: {
                        "type": fdef.data_type.value,
                        "importance": fdef.importance.name,
                        "description": fdef.description,
                    }
                    for fname, fdef in schema.fields.items()
                },
            }
        return {"domains": domains}

    @app.get("/v1/analytics")
    async def get_analytics():
        samples = _ANALYTICS["latency_samples"]
        avg_lat = sum(samples) / len(samples) if samples else 0
        return {
            "total_queries": _ANALYTICS["total_queries"],
            "domain_distribution": dict(_ANALYTICS["domain_counts"]),
            "conflict_rate": (
                round(_ANALYTICS["conflict_count"] / max(_ANALYTICS["total_queries"], 1), 3)
            ),
            "relaxation_rate": (
                round(_ANALYTICS["relaxation_count"] / max(_ANALYTICS["total_queries"], 1), 3)
            ),
            "avg_latency_ms": round(avg_lat, 2),
            "active_sessions": len(_SESSION_STORE),
        }

    # Serve static dashboard files
    if _DASHBOARD_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR)), name="static")


# ── Stdlib fallback server ─────────────────────────────────────────────

else:
    import http.server
    import socketserver

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.info(format, *args)

        def do_GET(self):
            if self.path == "/health":
                self._json({"status": "ok", "note": "Install fastapi for full API support"})
            elif self.path in ("/", "/dashboard"):
                index = _DASHBOARD_DIR / "index.html"
                if index.exists():
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(index.read_bytes())
                else:
                    self._json({"error": "Dashboard not found"}, 404)
            else:
                self._json({"error": "Not found. Install fastapi for full REST API."}, 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            query = body.get("query", "")
            parser_name = body.get("parser", "spacy")
            try:
                pipeline = _get_pipeline(parser_name)
                result = pipeline.run(query)
                self._json(result.to_dict())
            except Exception as e:
                self._json({"error": str(e)}, 500)

        def _json(self, data: dict, code: int = 200):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    app = None  # no FastAPI app in fallback mode


# ── Entry point ────────────────────────────────────────────────────────


def run(host: str = "0.0.0.0", port: int = 8080):
    if _FASTAPI:
        import uvicorn

        logger.info("Starting Prefilter AI Platform (FastAPI) on http://%s:%d", host, port)
        logger.info("Dashboard: http://localhost:%d/", port)
        logger.info("API docs:  http://localhost:%d/docs", port)
        uvicorn.run(app, host=host, port=port)
    else:
        logger.info(
            "FastAPI not installed. Starting stdlib fallback server on http://%s:%d", host, port
        )
        logger.info("Install FastAPI for full REST API: pip install fastapi uvicorn")
        with socketserver.TCPServer((host, port), _Handler) as httpd:
            httpd.serve_forever()


if __name__ == "__main__":
    run()
