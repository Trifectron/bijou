"""The skill bank in the agent's process: every trained skill loaded once, equipped per request."""

import pytest

pytest.importorskip("torch")

from engine.clients.local_bank import LocalBank  # noqa: E402
from engine.core.types.agent import PhaseSpec, RequestContext, SkillRequest  # noqa: E402
from engine.core.types.errors import SkillRuntimeError  # noqa: E402
from engine.runtime.bank import SkillBank  # noqa: E402
from engine.runtime.train import train_adapter  # noqa: E402
from engine.telemetry.metrics import BankMetrics, new_registry  # noqa: E402


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
def runtime(bank):
    return LocalBank(bank.cfg, bank=bank)


def context():
    return RequestContext.start("s", 30)


async def test_the_catalog_names_what_is_trained(runtime):
    skills = await runtime.catalog()
    assert [s.name for s in skills if s.trained] == ["json_extract"]
    assert "JSON" in next(s for s in skills if s.name == "json_extract").description


async def test_generation_with_a_skill_equipped_and_with_none(runtime):
    equipped = SkillRequest(prompt="Ana is an engineer.", skills=["json_extract"])
    assert (await runtime.run(context(), equipped)).skills == ["json_extract"]
    assert (await runtime.run(context(), SkillRequest(prompt="hello"))).skills == []


async def test_a_phase_schedule_is_accepted_when_it_aligns_with_blocks(runtime):
    schedule = [PhaseSpec(start=0.0, end=0.5, skills={"json_extract": 1.0})]
    await runtime.run(context(), SkillRequest(prompt="x", schedule=schedule))


@pytest.mark.parametrize(
    ("request_", "reason"),
    [
        (SkillRequest(prompt="x", skills=["nope"]), "not trained"),
        (SkillRequest(prompt="x", skills=["json_extract"], schedule=[]), "not both"),
        (
            SkillRequest(
                prompt="x", schedule=[PhaseSpec(start=0.0, end=0.3, skills={"json_extract": 1})]
            ),
            "block",
        ),
        (SkillRequest(prompt="x", gen_length=100000), "max_gen_length"),
    ],
)
async def test_untrained_skills_and_bad_requests_are_refused_with_the_reason(
    runtime, request_, reason
):
    with pytest.raises(SkillRuntimeError, match=reason):
        await runtime.run(context(), request_)


async def test_the_bank_counts_what_it_generates(bank, make_backend):
    registry = new_registry()
    counted = SkillBank(bank.cfg, backend=make_backend(bank.cfg), metrics=BankMetrics(registry))
    runtime = LocalBank(counted.cfg, bank=counted)
    await runtime.run(context(), SkillRequest(prompt="Ana", skills=["json_extract"]))
    with pytest.raises(SkillRuntimeError):
        await runtime.run(context(), SkillRequest(prompt="x", skills=["nope"]))
    sample = registry.get_sample_value
    assert sample("bijou_bank_generations_total", {"outcome": "ok", "skills": "json_extract"}) == 1
    assert sample("bijou_bank_generations_total", {"outcome": "error", "skills": "nope"}) == 1
    assert sample("bijou_bank_trained_skills", {}) == 1
    assert sample("bijou_bank_lock_wait_seconds_count", {}) == 1
