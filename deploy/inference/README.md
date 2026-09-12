# deploy/inference

The agent's chat model: `llama-server` from llama.cpp, speaking the OpenAI-compatible API.
Default GGUF `Qwen/Qwen3-4B-GGUF:Q4_K_M` on port 8000. `--jinja` applies the model's chat
template, which tool calls need; `--metrics` exports Prometheus format at `/metrics`.

## On the host

```bash
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --host 127.0.0.1 --port 8000 \
  --jinja --metrics -c 8192 -np 2 -ngl 99
```

## In compose

```bash
just up chat
```

GGUFs download on first run into the `modelcache` volume. Override with `BIJOU_CHAT_GGUF`,
`BIJOU_CHAT_CTX`, `BIJOU_CHAT_PARALLEL` and `BIJOU_CHAT_NGL`. Inside compose the host is `chat`.

`-c` (`--ctx-size`) is the total across slots: each of `-np` (`--parallel`) N slots gets
`ctx-size / N`.

## Pointing the engine at it

The engine reads `BIJOU_AGENT__LLM__BASE_URL`, default `http://127.0.0.1:8000/v1`:

```bash
BIJOU_AGENT__LLM__BASE_URL=http://127.0.0.1:8000/v1 just chat
```

## VRAM on a 6 GB GPU

The 4B Q4_K_M chat model takes ~3.1 GB: 2.5 GB of weights plus ~1.2 GB of KV cache at
`-c 8192 -np 2`. The in-process skill bank (150M, bf16) needs ~1-2 GB alongside it. Both fit;
raise `-c` only with headroom, and lower `-ngl` to spill layers to CPU.

## Metrics

```bash
just up phoenix prometheus grafana gpu-exporter
```

Grafana at http://localhost:3000 (admin/admin) with the `Bijou` dashboard (agent, skill bank,
llama-server, GPU); Prometheus at http://localhost:9090; Phoenix traces at http://localhost:6006.
Prometheus runs on the host network and scrapes loopback ports, whether a service runs on the
host or in compose with its port published there.
