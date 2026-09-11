"""Reading and writing one adapter's weights."""

from __future__ import annotations

from pathlib import Path

import torch

from engine.adapters.lora import add, sites
from engine.core.types.diffusion import AdapterSpec
from engine.core.types.errors import AdapterError


def state_dict(model: torch.nn.Module, name: str) -> dict[str, torch.Tensor]:
    """One adapter's weights, keyed by injection site."""
    out: dict[str, torch.Tensor] = {}
    for site_name, site in sites(model):
        if name not in site.names():
            continue
        for key, value in site.deltas[name].state_dict().items():
            out[f"{site_name}.{key}"] = value.detach().cpu().clone()
    if not out:
        raise AdapterError(f"no adapter {name} on this model")
    return out


def save(model: torch.nn.Module, name: str, path: Path, spec: AdapterSpec) -> None:
    """Write weights and the spec that produced them to one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec, "weights": state_dict(model, name)}, path)


def load(model: torch.nn.Module, path: Path, name: str | None = None) -> AdapterSpec:
    """Attach the adapter in path, under its own name unless one is given."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    spec: AdapterSpec = blob["spec"]
    if name and name != spec.name:
        spec = AdapterSpec(
            name=name, rank=spec.rank, alpha=spec.alpha, targets=spec.targets, dropout=spec.dropout
        )
    if not any(spec.name in site.names() for _, site in sites(model)):
        add(model, spec)
    for site_name, site in sites(model):
        prefix = f"{site_name}."
        weights = {k[len(prefix) :]: v for k, v in blob["weights"].items() if k.startswith(prefix)}
        if not weights:
            raise AdapterError(f"checkpoint has no weights for site {site_name}")
        site.deltas[spec.name].load_state_dict(weights)
    return spec
