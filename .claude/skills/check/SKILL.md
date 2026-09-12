---
name: check
description: Run the repo gate and fix what it reports. Use before claiming any change is done, and whenever a commit is about to be made.
---

# check

```
just check
```

Runs, in order: `ruff format --check`, `ruff check`, `scripts/check-deps.sh` (the import-linter
contracts), `mypy`, `pytest -m "not gpu"`.

Fix at the source. A `noqa` or a `type: ignore` gets the narrowest possible scope and a one-line
reason, and only when the constraint comes from a dependency's fixed signature.

A layering failure means either the import is wrong or the rule changed. If the rule changed, edit
the contracts in `pyproject.toml` and `docs/ARCHITECTURE.md` in the same commit. Never widen a
contract to make a red build green.

`just test gpu` is separate and needs a GPU and a base checkpoint. It is not part of the gate.
