"""The configuration: what loads, what the agent reads, and what is rejected at load."""

import pytest

from engine.core.config import AgentConfig, Config, Telemetry
from engine.core.types.errors import ConfigError


def test_defaults_load_and_only_harness_keys_are_read(tmp_path, monkeypatch):
    path = tmp_path / "bijou.toml"
    path.write_text("[train]\nlr = 1\n[agent.loop]\nmax_turns = 3\n[agent.llm]\nmodel = 'm'\n")
    monkeypatch.setenv("BIJOU_CONFIG_FILE", str(path))
    monkeypatch.setenv("BIJOU_AGENT__LLM__API_KEY", "secret")
    cfg = Config().agent
    assert cfg.loop.max_turns == 3 and cfg.llm.model == "m"
    assert cfg.llm.api_key.get_secret_value() == "secret"
    assert "secret" not in cfg.model_dump_json()


def test_the_committed_file_loads():
    cfg = Config()
    assert cfg.bank.max_gen_length > 0 and cfg.telemetry.metrics_port > 0


@pytest.mark.parametrize(
    ("table", "values", "message"),
    [
        ("loop", {"max_turns": 0}, "max_turns"),
        ("planning", {"concurrency": 0}, "concurrency"),
        ("patterns", {"similarity": 0}, "similarity"),
        ("policy", {"confirm_from": "read_public"}, "every tool"),
    ],
)
def test_contradictions_are_rejected_at_load(table, values, message):
    with pytest.raises(ConfigError, match=message):
        AgentConfig(**{table: values})


def test_the_metrics_port_is_a_tcp_port_or_none():
    assert Telemetry(metrics_port=0).metrics_port == 0
    with pytest.raises(ConfigError, match="metrics_port"):
        Telemetry(metrics_port=70000)


def test_two_mcp_servers_cannot_share_a_name():
    with pytest.raises(ConfigError, match="two servers named pages"):
        AgentConfig(
            mcp={
                "servers": [
                    {"name": "pages", "url": "http://x"},
                    {"name": "pages", "command": "y"},
                ]
            }
        )
