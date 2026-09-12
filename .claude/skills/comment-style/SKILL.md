---
name: comment-style
description: The comment rules for this repo. Use when writing or editing any Python comment or docstring, and when reviewing a diff that touches comments.
---

# Comment style

Applies to every `#` and `"""` under `apps/`, tests included, for new code and for any comment a change
touches. `third_party/` is exempt; it is never edited.

## Say what the code does

A comment states behaviour. It does not argue for it.

No rationale, trade-offs, history, alternatives considered, or what a past version did. Those
belong in the commit message.

```python
# Rows are scored one block at a time.
CHUNK = 1024
```

Not:

```python
# Chunked rather than one buffer for the whole collection, because that buffer would be the size
# of the data and thrown away once per query.
CHUNK = 1024
```

An invariant the code depends on is kept as a fact, without the argument for it.

## Plain ASCII, monotone

No backticks, no quotation marks around terms, no em dashes, no arrows, no non-ASCII. Present
tense, declarative, one line where one line does.

No openers (Note that, Simply, Basically, Crucially, Importantly) and no closing summary.

## Docstrings

Every public function, class and module has a one-line docstring saying what, not how. A module
docstring may add a short paragraph when the module's role in the package is not obvious from its
name.

Args and Returns sections only where the types do not already say it.
