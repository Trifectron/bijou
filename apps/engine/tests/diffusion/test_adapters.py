"""Adapter injection. Marked gpu only where a base checkpoint is needed.

Step 0 of docs/ROADMAP.md lives here: an injected model with an untrained
adapter active must produce bit-identical output to the model without one.
"""

import pytest

torch = pytest.importorskip("torch")

from engine.adapters.io import load, save, state_dict  # noqa: E402
from engine.adapters.lora import LoRALinear, add, inject, sites, trainable  # noqa: E402
from engine.core.types.diffusion import AdapterSpec  # noqa: E402
from engine.core.types.errors import AdapterError  # noqa: E402

SPEC = AdapterSpec(name="probe", rank=4, alpha=4.0, targets=("attn.qkv",))


def _ids(model):
    return torch.randint(0, 100, (1, 16))


def test_injection_wraps_the_named_linears(tiny_model):
    inject(tiny_model, ("attn.qkv",))
    names = [n for n, _ in sites(tiny_model)]
    assert names and all(n.endswith("attn.qkv") for n in names)


def test_injection_refuses_unmatched_targets(tiny_model):
    with pytest.raises(AdapterError, match="no Linear matched"):
        inject(tiny_model, ("does.not.exist",))


def test_targets_match_the_qualified_name_not_the_leaf(tiny_model):
    with pytest.raises(AdapterError, match="no Linear matched"):
        inject(tiny_model, ("mlp.qkv",))


def test_double_injection_is_refused(tiny_model):
    inject(tiny_model, ("attn.qkv",))
    with pytest.raises(AdapterError, match="already injected"):
        inject(tiny_model, ("attn.qkv",))


def test_untrained_adapter_is_inert(tiny_model):
    ids = _ids(tiny_model)
    tiny_model.eval()
    with torch.no_grad():
        before = tiny_model(ids)
    state = inject(tiny_model, ("attn.qkv", "mlp.w1"))
    add(tiny_model, SPEC)
    state.set("probe")
    with torch.no_grad():
        after = tiny_model(ids)
    assert torch.equal(before, after)


def test_trained_adapter_changes_output(tiny_model):
    ids = _ids(tiny_model)
    tiny_model.eval()
    state = inject(tiny_model, ("attn.qkv",))
    add(tiny_model, SPEC)
    for _, site in sites(tiny_model):
        torch.nn.init.normal_(site.deltas["probe"].B.weight, std=0.1)
    with torch.no_grad():
        state.clear()
        off = tiny_model(ids)
        state.set("probe")
        on = tiny_model(ids)
    assert not torch.equal(off, on)


def test_weights_scale_the_delta(tiny_model):
    ids = _ids(tiny_model)
    tiny_model.eval()
    state = inject(tiny_model, ("attn.qkv",))
    add(tiny_model, SPEC)
    for _, site in sites(tiny_model):
        torch.nn.init.normal_(site.deltas["probe"].B.weight, std=0.1)
    with torch.no_grad():
        state.set("probe")
        full = tiny_model(ids)
        state.active = {"probe": 0.0}
        zero = tiny_model(ids)
        state.clear()
        off = tiny_model(ids)
    assert torch.equal(zero, off)
    assert not torch.equal(full, off)


def test_only_the_adapter_trains(tiny_model):
    inject(tiny_model, ("attn.qkv",))
    add(tiny_model, SPEC)
    count = trainable(tiny_model, "probe")
    assert count > 0
    assert count == sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    assert all(
        not m.base.weight.requires_grad for m in tiny_model.modules() if isinstance(m, LoRALinear)
    )


def test_adapter_round_trips(tiny_model, tmp_path):
    inject(tiny_model, ("attn.qkv",))
    add(tiny_model, SPEC)
    for _, site in sites(tiny_model):
        torch.nn.init.normal_(site.deltas["probe"].B.weight, std=0.1)
    path = tmp_path / "probe.pt"
    save(tiny_model, "probe", path, SPEC)
    original = state_dict(tiny_model, "probe")

    from nanodiff.model import NanoDiff

    from engine.backends.nanodiff import tiny_config

    torch.manual_seed(0)
    other = NanoDiff(tiny_config())
    inject(other, ("attn.qkv",))
    spec = load(other, path)
    assert spec.name == "probe"
    reloaded = state_dict(other, "probe")
    assert all(torch.equal(original[k], reloaded[k]) for k in original)
