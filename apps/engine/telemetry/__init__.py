"""What the engine reports about itself.

  metrics  Prometheus counters and histograms for the agent and the skill bank, served at /metrics
  otel     OpenTelemetry spans built from trace events, exported over OTLP to Phoenix

Both read the trace events the agent already emits, so neither changes the loop. Each registry
and tracer is owned by the object that built it; nothing here is process-global.
"""
