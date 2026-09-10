# The training image. CPU checks run in CI without it; this is for a GPU box.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml bijou.toml ./
RUN uv sync --extra train

COPY bijou ./bijou
COPY third_party ./third_party

ENTRYPOINT ["uv", "run", "bijou"]
CMD ["--help"]
