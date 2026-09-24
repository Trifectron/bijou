"""The skill bank: one base model with every trained adapter attached, equipped per request.

Adapters are loaded once at start. A request names what to equip, and equipping is a PhaseRouter
over the shared AdapterState, so no request reloads weights. Generation holds a lock: the model
and its AdapterState are one piece of mutable state, and one GPU runs one generation at a time.
"""

from __future__ import annotations

import threading
import time

from engine.core.config import Config
from engine.core.protocols import Backend
from engine.core.types.agent import SkillInfo, SkillRequest, SkillResult
from engine.core.types.diffusion import GenerationRequest
from engine.core.types.errors import AdapterError, ConfigError, EngineError
from engine.routing.phase import Phase, PhaseRouter, PhaseSchedule
from engine.runtime.evaluate import prepare
from engine.skills import load as load_skill
from engine.skills import names as skill_names
from engine.telemetry.metrics import BankMetrics, skills_label


class SkillBank:
    """The base model and its trained skills, ready to generate."""

    def __init__(
        self,
        cfg: Config,
        backend: Backend | None = None,
        metrics: BankMetrics | None = None,
    ) -> None:
        loading = time.monotonic()
        self.cfg = cfg
        self.metrics = metrics
        self.known = skill_names(cfg.paths.data)
        self.trained = tuple(n for n in self.known if cfg.adapter_path(n).exists())
        self.backend, self.state = prepare(cfg, list(self.trained), backend=backend)
        self._lock = threading.Lock()
        if metrics is not None:
            metrics.trained_skills.set(len(self.trained))
            metrics.load_seconds.set(time.monotonic() - loading)

    def catalog(self) -> list[SkillInfo]:
        """Every known skill, with its description and whether it can be equipped."""
        return [
            SkillInfo(
                name=name,
                description=load_skill(name, self.cfg.paths.data).DESCRIPTION,
                trained=name in self.trained,
            )
            for name in self.known
        ]

    def schedule(self, req: SkillRequest) -> PhaseSchedule:
        """The request as a schedule over trained skills. Rejects anything else."""
        if req.schedule is not None and req.skills:
            raise ConfigError("give skills or a schedule, not both")
        if req.schedule is None:
            schedule = PhaseSchedule.static(*req.skills)
        else:
            if not req.schedule:
                raise ConfigError("an empty schedule equips nothing; send skills=[] instead")
            schedule = PhaseSchedule(*(Phase(p.start, p.end, dict(p.skills)) for p in req.schedule))
        missing = sorted({n for p in schedule.phases for n in p.adapters if n not in self.trained})
        if missing:
            raise AdapterError(
                f"not trained: {', '.join(missing)}; trained: {', '.join(self.trained) or 'none'}"
            )
        return schedule

    def request(self, req: SkillRequest) -> GenerationRequest:
        """The sampler settings: the configured ones, overridden within the server's limits."""
        sampling = self.cfg.sampling
        gen_length = req.gen_length or sampling.gen_length
        steps = req.steps or sampling.steps
        block = min(sampling.block_length, gen_length)
        if gen_length > self.cfg.bank.max_gen_length:
            raise ConfigError(f"gen_length {gen_length} exceeds bank.max_gen_length")
        if gen_length % block or steps % (gen_length // block):
            raise ConfigError(
                f"gen_length {gen_length} and steps {steps} do not split into blocks of {block}"
            )
        return GenerationRequest(
            prompt=self.prompt(req),
            gen_length=gen_length,
            steps=steps,
            block_length=block,
            temperature=sampling.temperature,
        )

    def prompt(self, req: SkillRequest) -> str:
        """The prompt, inside the first equipped skill's template when instruct is set."""
        equipped = req.skills or [n for p in req.schedule or [] for n in p.skills]
        if not req.instruct or not equipped:
            return req.prompt
        skill = load_skill(equipped[0], self.cfg.paths.data)
        return skill.PROMPT.format(instruction=skill.INSTRUCTIONS[0], text=req.prompt)

    def generate(self, req: SkillRequest) -> SkillResult:
        """Equip, generate, and restore the previous equipped set."""
        asked = req.skills or [n for p in req.schedule or [] for n in p.skills]
        try:
            schedule = self.schedule(req)
            gen = self.request(req)
            blocks = gen.gen_length // (gen.block_length or gen.gen_length)
            schedule.validate_against_blocks(gen.steps, blocks)
        except EngineError:
            if self.metrics is not None:
                self.metrics.generations.labels(skills_label(asked), "error").inc()
            raise
        live = sorted({n for p in schedule.phases for n in p.adapters})
        waiting = time.monotonic()
        with self._lock:
            started = time.monotonic()
            with PhaseRouter(self.state, schedule) as router:
                text = self.backend.generate(gen, on_step=router.at)
        if self.metrics is not None:
            label = skills_label(live)
            self.metrics.lock_wait_seconds.observe(started - waiting)
            self.metrics.generation_seconds.labels(label).observe(time.monotonic() - started)
            self.metrics.generations.labels(label, "ok").inc()
            self.metrics.tokens.inc(gen.gen_length)
        return SkillResult(
            text=text,
            skills=live,
            gen_length=gen.gen_length,
            steps=gen.steps,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
