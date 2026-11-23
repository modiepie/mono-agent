# One image for the API and the UI (different commands in docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Dependencies first: this layer is cached until uv.lock changes. The uv download cache is
# a BuildKit cache mount, so it speeds up rebuilds without ending up in the image. On Linux torch comes from
# the CPU-only index (pyproject [tool.uv.sources]), so no CUDA libraries end up in the image.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data/docs ./data/docs
COPY .streamlit ./.streamlit
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# Unprivileged user: the app needs no root inside the container.
RUN useradd --create-home app && mkdir -p /app/state && chown app /app/state
USER app

EXPOSE 8000 8501
CMD ["mono-agent-api"]
