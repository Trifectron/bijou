# The engine image: apps/engine with its model stack, for a GPU box. engine serve runs the agent
# with the skill bank in process; engine serve-skills runs the bank alone. evals and the console
# run from source.
# TORCH selects the torch build: cuda, or cpu for an image that runs without a GPU.
FROM ghcr.io/astral-sh/uv:0.11-python3.12-trixie-slim

ARG TORCH=cuda
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
# Every member's pyproject.toml, so the workspace lockfile can be checked.
COPY pyproject.toml uv.lock ./
COPY apps/engine/pyproject.toml apps/engine/
COPY apps/evals/pyproject.toml apps/evals/
COPY apps/cli/pyproject.toml apps/cli/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-workspace --package engine --extra train --extra ${TORCH}

COPY bijou.toml ./
COPY apps/engine ./apps/engine
COPY third_party ./third_party
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package engine --extra train --extra ${TORCH}

ARG GIT_SHA=unknown
ENV BIJOU_GIT_SHA=${GIT_SHA}

# Both servers bind loopback by default; set BIJOU_AGENT__HTTP__HOST or BIJOU_SERVE__HOST to
# 0.0.0.0 to reach them from outside.
EXPOSE 8200 8100
ENTRYPOINT ["engine"]
CMD ["--help"]
