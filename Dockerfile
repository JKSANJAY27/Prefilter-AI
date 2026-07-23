# Multi-stage Dockerfile for Prefilter AI Platform API Server & Dashboard

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package manifests
COPY pyproject.toml README.md /app/

# Copy codebase
COPY prefilter_ai /app/prefilter_ai
COPY prefilter_platform /app/prefilter_platform

# Install python dependencies including platform extra & spaCy model
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[platform]" && \
    python -m spacy download en_core_web_sm

# Expose server port
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default launch command
CMD ["uvicorn", "prefilter_platform.server:app", "--host", "0.0.0.0", "--port", "8080"]
