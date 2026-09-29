# --- frontend build -----------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- API (+ built frontend) ---------------------------------------------------
FROM python:3.12-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# chromium lets kaleido render chart PNGs server-side (best effort; the UI can also export PNGs)
RUN apt-get update && apt-get install -y --no-install-recommends chromium && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 datasleuth
WORKDIR /app
COPY pyproject.toml README.md ./
COPY datasleuth ./datasleuth
COPY eval ./eval
RUN pip install .
COPY --from=web /web/dist ./frontend/dist
ENV DATASLEUTH_DATA_DIR=/data DATASLEUTH_FRONTEND_DIST=/app/frontend/dist DATASLEUTH_CACHE_DIR=/data/cache/llm DATASLEUTH_LOG_DIR=/data/logs
RUN mkdir -p /data && chown -R datasleuth /data
USER datasleuth
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["datasleuth", "serve", "--host", "0.0.0.0", "--port", "8000"]
