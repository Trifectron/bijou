"""The skill bank: every trained skill loaded once, equipped per request, remote or local."""

import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from engine.api.bank import create_app  # noqa: E402
from engine.clients.local_bank import LocalBank  # noqa: E402
from engine.core.types.agent import RequestContext, SkillRequest  # noqa: E402
from engine.core.types.errors import SkillRuntimeError  # noqa: E402
from engine.runtime.bank import SkillBank  # noqa: E402
from engine.runtime.train import train_adapter  # noqa: E402


@pytest.fixture
def bank(cfg, tmp_path, make_backend):
    local = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={
                    "adapters": tmp_path / "adapters",
                    "runs": tmp_path / "runs",
                    "data": tmp_path,
                }
            )
        }
    )
    train_adapter(local, "json_extract", backend=make_backend(local))
    return SkillBank(local, backend=make_backend(local))


@pytest.fixture
def client(bank):
    return TestClient(create_app(bank))


def test_health_and_the_catalog(client):
    assert client.get("/health").json()["trained"] == 1
    skills = client.get("/skills").json()
    assert skills == [
        {
            "name": "json_extract",
            "description": skills[0]["description"],
            "trained": True,
        }
    ]
    assert "JSON" in skills[0]["description"]


def test_generation_with_a_skill_equipped_and_with_none(client):
    equipped = client.post(
        "/generate", json={"prompt": "Ana is an engineer.", "skills": ["json_extract"]}
    )
    assert equipped.status_code == 200
    assert equipped.json()["skills"] == ["json_extract"]
    bare = client.post("/generate", json={"prompt": "hello"})
    assert bare.status_code == 200 and bare.json()["skills"] == []


def test_a_phase_schedule_is_accepted_when_it_aligns_with_blocks(client):
    body = {
        "prompt": "x",
        "schedule": [{"start": 0.0, "end": 0.5, "skills": {"json_extract": 1.0}}],
    }
    assert client.post("/generate", json=body).status_code == 200


def test_untrained_skills_and_bad_requests_are_rejected_with_the_reason(client):
    unknown = client.post("/generate", json={"prompt": "x", "skills": ["nope"]})
    assert unknown.status_code == 422 and "not trained" in unknown.json()["detail"]
    both = client.post(
        "/generate",
        json={"prompt": "x", "skills": ["json_extract"], "schedule": []},
    )
    assert both.status_code == 422
    misaligned = client.post(
        "/generate",
        json={
            "prompt": "x",
            "schedule": [{"start": 0.0, "end": 0.3, "skills": {"json_extract": 1}}],
        },
    )
    assert misaligned.status_code == 422 and "block" in misaligned.json()["detail"]
    too_long = client.post("/generate", json={"prompt": "x", "gen_length": 100000})
    assert too_long.status_code == 422


async def test_the_bank_runs_in_process_behind_the_same_protocol(bank):
    runtime = LocalBank(bank.cfg, bank=bank)
    ctx = RequestContext.start("s", 30)
    assert [s.name for s in await runtime.catalog() if s.trained] == ["json_extract"]
    result = await runtime.run(ctx, SkillRequest(prompt="Ana", skills=["json_extract"]))
    assert result.skills == ["json_extract"]
    with pytest.raises(SkillRuntimeError, match="not trained"):
        await runtime.run(ctx, SkillRequest(prompt="x", skills=["nope"]))


def test_the_bank_counts_what_it_generates(bank, make_backend):
    from engine.telemetry.metrics import BankMetrics, new_registry

    registry = new_registry()
    counted = SkillBank(bank.cfg, backend=make_backend(bank.cfg), metrics=BankMetrics(registry))
    client = TestClient(create_app(counted, registry))
    client.post("/generate", json={"prompt": "Ana", "skills": ["json_extract"]})
    client.post("/generate", json={"prompt": "x", "skills": ["nope"]})
    body = client.get("/metrics").text
    assert 'bijou_bank_generations_total{outcome="ok",skills="json_extract"} 1.0' in body
    assert 'bijou_bank_generations_total{outcome="error",skills="nope"} 1.0' in body
    assert "bijou_bank_trained_skills 1.0" in body
    assert "bijou_bank_lock_wait_seconds_count 1.0" in body
