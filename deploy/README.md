# Deploy

Bijou runs one-shot jobs, not services: train an adapter, score the matrix, exit. Deploying means
getting the repo, a base checkpoint and a GPU into the same place.

## On a GPU box, from source

```bash
git clone --recurse-submodules https://github.com/Trifectron/bijou && cd bijou
just bootstrap        # .env, hooks, submodule, dependencies without torch
just setup-train      # CUDA torch, pinned by uv.lock
just checkpoints      # the base checkpoint named by backend.checkpoint in bijou.toml
just doctor           # every line should read ok
just matrix           # train every configured skill, then score the matrix
```

`just checkpoints nanodiff-50m-sft-alpaca nanodiff-350m-base` fetches others by name. The six
published checkpoints are listed in `third_party/nanoDiff/README.md`.

## The training image

CD builds `ghcr.io/trifectron/bijou/training` on every push to `main` that changes what the image
contains, tagged with the branch, the short SHA, and a version on `v*` tags. The image carries the
code, `bijou.toml` and the locked dependencies. Checkpoints, adapters and run records are mounted.

```bash
docker run --rm --gpus all \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/runs:/app/runs" \
  ghcr.io/trifectron/bijou/training:main skill train json_extract
```

Override any setting with `-e BIJOU_<SECTION>__<KEY>=value`. Run records written from the image
carry the commit it was built from, through `BIJOU_GIT_SHA`.

`just image` builds the same image locally; `just image cpu` builds one with CPU torch, which is
how to check the Dockerfile on a machine without a GPU.
