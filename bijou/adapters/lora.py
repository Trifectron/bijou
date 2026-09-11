"""LoRA adapters.

The base models are 50M to 350M, where LoRA saves no memory. It is used for
modularity: a named, detachable delta that trains in isolation and sums with
other deltas. Whether those properties survive summation is what the composition
experiment measures.

Targets are named by module suffix and configured in bijou.toml. lm_head is
rejected in config because it is weight-tied to tok_emb.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch
import torch.nn as nn

from bijou.core.types import AdapterError, AdapterSpec, AdapterState


class _Delta(nn.Module):
    """One adapter's low-rank branch: x to B(A(x)) times alpha over rank."""

    def __init__(self, in_features: int, out_features: int, spec: AdapterSpec) -> None:
        super().__init__()
        self.A = nn.Linear(in_features, spec.rank, bias=False)
        self.B = nn.Linear(spec.rank, out_features, bias=False)
        self.scaling = spec.scaling
        self.drop = nn.Dropout(spec.dropout) if spec.dropout else nn.Identity()
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.drop(x))) * self.scaling


class LoRALinear(nn.Module):
    """A frozen Linear with a name-keyed bank of deltas. Satisfies AdapterSite."""

    def __init__(self, base: nn.Linear, state: AdapterState) -> None:
        super().__init__()
        self.base = base
        self.state = state
        self.deltas = nn.ModuleDict()

    def add(self, spec: AdapterSpec) -> None:
        if spec.name in self.deltas:
            raise AdapterError(f"adapter {spec.name} is already attached")
        delta = _Delta(self.base.in_features, self.base.out_features, spec)
        self.deltas[spec.name] = delta.to(self.base.weight.device, dtype=self.base.weight.dtype)

    def names(self) -> Sequence[str]:
        return tuple(self.deltas.keys())

    def parameters_for(self, name: str) -> Iterator[nn.Parameter]:
        if name not in self.deltas:
            raise AdapterError(f"no adapter {name} at this site")
        return self.deltas[name].parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for name, weight in self.state.active.items():
            if weight and name in self.deltas:
                out = out + weight * self.deltas[name](x)
        return out


def inject(
    model: nn.Module, targets: Sequence[str], state: AdapterState | None = None
) -> AdapterState:
    """Wrap every Linear whose qualified name ends in a target. Call once."""
    state = state or AdapterState()
    wrapped = 0
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                raise AdapterError("model is already injected")
            if not isinstance(child, nn.Linear):
                continue
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(qualified == t or qualified.endswith(f".{t}") for t in targets):
                setattr(parent, child_name, LoRALinear(child, state))
                wrapped += 1
    if not wrapped:
        raise AdapterError(f"no Linear matched targets {tuple(targets)}")
    return state


def sites(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    """Every injection site, by qualified name."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def add(model: nn.Module, spec: AdapterSpec) -> None:
    """Attach a zero-initialised, therefore inert, adapter at every site."""
    found = sites(model)
    if not found:
        raise AdapterError("no injection sites, call inject first")
    for _, site in found:
        site.add(spec)


def trainable(model: nn.Module, name: str) -> int:
    """Freeze everything, unfreeze one adapter, return its parameter count."""
    for p in model.parameters():
        p.requires_grad_(False)
    count = 0
    for _, site in sites(model):
        if name not in site.names():
            continue
        for p in site.parameters_for(name):
            p.requires_grad_(True)
            count += p.numel()
    if not count:
        raise AdapterError(f"no trainable parameters for adapter {name}")
    return count
