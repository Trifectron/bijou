"""LoRA adapters for nanoDiff.

Why LoRA here, given the base models are 50M-350M and full fine-tuning is
affordable? Not for memory. The point is *modularity*: an adapter is a named,
detachable delta that can be trained in isolation, swapped at runtime, and
summed with other deltas. Whether those properties actually survive summation
is Bijou's central empirical question -- see docs/experiments.md.

Design notes specific to nanoDiff (third_party/nanoDiff @ 312a9e7):

  * Target modules are `attn.qkv`, `attn.proj`, `mlp.w1`, `mlp.w2`, `mlp.w3`.
    `qkv` is a single fused (n_embd -> 3*n_embd) Linear; one LoRA over the
    fused matrix is the standard treatment and is what we do.

  * `lm_head` is DELIBERATELY not a target. `config.tie_embeddings=True` ties
    `lm_head.weight` to `tok_emb.weight`, so adapting the head silently adapts
    the embedding table.

  * Activation is driven by a single shared `AdapterState` object rather than
    by walking modules and flipping flags. Switching the active set is then
    O(1), which is what makes per-denoising-step phase routing cheap enough to
    do inside the sampler loop (see bijou/phase.py).

  * nanoDiff's `configure_optimizers` already filters on `requires_grad`, so
    `freeze_base(model)` is sufficient to make it an adapter-only optimizer.

NOTE: written against the upstream source but NOT executed -- this container
has no torch. Run `pytest tests/` on a machine with the nanoDiff env before
trusting any of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("attn.qkv", "attn.proj", "mlp.w1", "mlp.w2", "mlp.w3")


@dataclass
class AdapterState:
    """Which adapters are live, and at what weight.

    Shared by reference across every LoRALinear in a model. Mutating `active`
    changes the model's behaviour immediately, with no module traversal.

    `active` maps adapter name -> scalar multiplier. A multiplier of 1.0 is the
    adapter at its trained strength; 0.7/0.4 style blends are the composition
    experiment (RQ2).
    """

    active: dict[str, float] = field(default_factory=dict)

    def set(self, *names: str, **weighted: float) -> None:
        """Replace the active set. `set("a", b=0.5)` -> {"a": 1.0, "b": 0.5}."""
        self.active = {n: 1.0 for n in names} | dict(weighted)

    def clear(self) -> None:
        self.active = {}


class _LoRABranch(nn.Module):
    """One adapter's low-rank delta: x -> B(A(x)) * (alpha / r)."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float, dropout: float = 0.0):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # B=0 makes the adapter an exact no-op at init, so attaching an
        # untrained adapter cannot perturb the frozen base model.
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """Frozen nn.Linear plus a name-keyed bank of low-rank branches."""

    def __init__(self, base: nn.Linear, state: AdapterState):
        super().__init__()
        self.base = base
        self.state = state
        self.adapters = nn.ModuleDict()

    def add_adapter(self, name: str, rank: int, alpha: float, dropout: float = 0.0):
        if name in self.adapters:
            raise KeyError(f"adapter {name!r} already attached")
        branch = _LoRABranch(self.base.in_features, self.base.out_features,
                             rank, alpha, dropout)
        self.adapters[name] = branch.to(self.base.weight.device,
                                        dtype=self.base.weight.dtype)

    def forward(self, x):
        out = self.base(x)
        for name, weight in self.state.active.items():
            branch = self.adapters.get(name)
            if branch is not None and weight != 0.0:
                out = out + weight * branch(x)
        return out


def _iter_targets(model: nn.Module, targets):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(full.endswith(t) for t in targets):
                yield module, child_name, child


def inject(model: nn.Module, targets=DEFAULT_TARGETS,
           state: AdapterState | None = None) -> AdapterState:
    """Wrap every target Linear in a LoRALinear. Idempotent-unsafe: call once."""
    state = state or AdapterState()
    for parent, child_name, child in _iter_targets(model, targets):
        setattr(parent, child_name, LoRALinear(child, state))
    return state


def add_adapter(model: nn.Module, name: str, rank: int = 16,
                alpha: float | None = None, dropout: float = 0.0) -> None:
    """Attach a new (zero-initialised, therefore no-op) adapter everywhere."""
    alpha = float(rank) if alpha is None else alpha
    sites = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if not sites:
        raise RuntimeError("no LoRALinear sites -- call inject(model) first")
    for site in sites:
        site.add_adapter(name, rank, alpha, dropout)


def freeze_base(model: nn.Module, name: str | None = None) -> int:
    """Freeze everything, then unfreeze adapter `name` (or all adapters).

    Returns the number of trainable parameters, so training scripts can assert
    they are optimising what they think they are.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for site in model.modules():
        if not isinstance(site, LoRALinear):
            continue
        for adapter_name, branch in site.adapters.items():
            if name is not None and adapter_name != name:
                continue
            for p in branch.parameters():
                p.requires_grad_(True)
                trainable += p.numel()
    return trainable


def adapter_state_dict(model: nn.Module, name: str) -> dict[str, torch.Tensor]:
    """Extract one adapter's weights, keyed by its injection site."""
    out = {}
    for site_name, site in model.named_modules():
        if isinstance(site, LoRALinear) and name in site.adapters:
            for k, v in site.adapters[name].state_dict().items():
                out[f"{site_name}.{k}"] = v.detach().cpu().clone()
    if not out:
        raise KeyError(f"no adapter named {name!r}")
    return out


def load_adapter(model: nn.Module, name: str, sd: dict[str, torch.Tensor],
                 rank: int = 16, alpha: float | None = None) -> None:
    """Attach `name` if absent, then load `sd` into it."""
    if not any(isinstance(m, LoRALinear) and name in m.adapters
               for m in model.modules()):
        add_adapter(model, name, rank=rank, alpha=alpha)
    for site_name, site in model.named_modules():
        if not (isinstance(site, LoRALinear) and name in site.adapters):
            continue
        prefix = f"{site_name}."
        branch_sd = {k[len(prefix):]: v for k, v in sd.items()
                     if k.startswith(prefix)}
        if not branch_sd:
            raise KeyError(f"checkpoint has no weights for site {site_name!r}")
        site.adapters[name].load_state_dict(branch_sd)
