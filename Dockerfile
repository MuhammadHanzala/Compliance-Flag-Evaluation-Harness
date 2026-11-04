# syntax=docker/dockerfile:1

# Shared base: builder and runtime must use the same Python path for the venv to work.
FROM python:3.15-slim-trixie AS base
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --no-create-home nonroot
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /app


# Builder: locked prod deps first (cached until uv.lock changes), then the project.
FROM base AS builder
# pyyaml and zstandard have no cp315 wheels yet, so uv builds them from source.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable
COPY pyproject.toml uv.lock README.md ./
COPY evalharness/ evalharness/
# --reinstall-package: uv only rebuilds the project when pyproject.toml changes, so without it
# the cache mount can serve a wheel built from older .py files (this shipped stale code once).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --reinstall-package compliance-flag-eval


# Dev: adds the dev group and the whole repo; runs the same checks as CI.
FROM builder AS dev
ENV UV_NO_DEV=0
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked
# No cache or bytecode writes, so the checks run read-only as any user.
ENV HOME=/tmp \
    RUFF_NO_CACHE=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_ADDOPTS="-p no:cacheprovider"
USER nonroot
CMD ["sh", "-c", "ruff check . && ruff format --check . && pytest"]


# Runtime (default target): the venv plus the data the CLI reads. No uv, no tests, no key.
FROM base AS runtime
COPY --from=builder /app/.venv /app/.venv
COPY data/ data/
COPY prompts/ prompts/
COPY config/ config/
RUN install -d -o nonroot -g nonroot /app/runs /app/reports
# HOME=/tmp lets compose run the image as any host UID.
ENV HOME=/tmp
USER nonroot
ENTRYPOINT ["evalharness"]
CMD ["--help"]
