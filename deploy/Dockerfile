# The training image. CPU checks run in CI without it; this is for a GPU box.
# TORCH selects the torch build: cuda, or cpu for an image that runs without a GPU.
FROM ghcr.io/astral-sh/uv:0.11-python3.12-trixie-slim

ARG TORCH=cuda
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra train --extra ${TORCH}

COPY bijou.toml ./
COPY bijou ./bijou
COPY third_party ./third_party
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra train --extra ${TORCH}

ARG GIT_SHA=unknown
ENV BIJOU_GIT_SHA=${GIT_SHA}

ENTRYPOINT ["bijou"]
CMD ["--help"]
