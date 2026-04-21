# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy dependency files first — layer cache only invalidated when deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy full source and install project
COPY . .
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

WORKDIR /app

# Only curl for HEALTHCHECK — no build tools, no pip, no uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled venv from builder (no build tools needed at runtime)
COPY --from=builder /app/.venv          /app/.venv

# Copy only application source files needed at runtime
COPY --from=builder /app/server         /app/server
COPY --from=builder /app/tasks          /app/tasks
COPY --from=builder /app/models.py      /app/models.py
COPY --from=builder /app/inference.py   /app/inference.py
COPY --from=builder /app/openenv.yaml   /app/openenv.yaml
COPY --from=builder /app/README.md      /app/README.md
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Enable HuggingFace Spaces web UI — without this the Space shows a blank page
ENV ENABLE_WEB_INTERFACE=true

# Use venv binaries directly
ENV PATH="/app/.venv/bin:$PATH"

# Ensure Python finds project modules from /app
ENV PYTHONPATH="/app"

# Required for [START]/[STEP]/[END] log lines to flush immediately
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]