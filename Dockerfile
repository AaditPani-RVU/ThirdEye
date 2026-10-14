# Third Eye server for Cloud Run: the FastAPI app (replays, specs, compile,
# history, the edge relay) plus the built web UI. No camera stack, no models,
# no GPU -- perception runs on the edge laptop.
#
#   docker build -t thirdeye .
#   docker run -p 8080:8080 -e GROQ_API_KEY=... thirdeye

# --- the web UI -------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# Same-origin build (no VITE_API_URL): the server below serves it. The Firebase
# web config is passed at build time when sign-in is on.
ARG VITE_FIREBASE_API_KEY=""
ARG VITE_FIREBASE_AUTH_DOMAIN=""
ARG VITE_FIREBASE_PROJECT_ID=""
ARG VITE_FIREBASE_APP_ID=""
RUN npm run build

# --- the server ---------------------------------------------------------------------
FROM python:3.12-slim
# OpenCV (a core dependency, used by replays) needs these shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY thirdeye/ thirdeye/
# Editable, so the package finds specs/ and scripts/ beside it in /app.
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir -e ".[server,cloud]"
COPY specs/ specs/
COPY scripts/scenarios/ scripts/scenarios/
COPY ui/ ui/
COPY --from=web /web/dist web/dist

ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8080 THIRDEYE_STORE=firestore THIRDEYE_AUTH=firebase
RUN useradd --create-home thirdeye && mkdir -p /tmp/sessions && chown thirdeye /tmp/sessions
USER thirdeye
EXPOSE 8080
CMD ["python", "-m", "thirdeye.server.app", "--logs", "/tmp/sessions"]
