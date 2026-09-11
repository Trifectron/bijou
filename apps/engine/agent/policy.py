"""The risk policy: allow, deny, or hold for the user, by a tool's risk class.

  read_public         run
  read_authenticated  deny unless policy.allow_authenticated_reads
  prepare_write       run, unless policy.confirm_from is lowered to it
  external_write      confirm immediately before
  destructive         confirm immediately before
  forbidden           deny, whatever the settings say

A confirmation is bound to one exact payload by hash, single use, and expires.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Any
from uuid import uuid4

from engine.core.config import Policy
from engine.core.types.agent import ConfirmationRequest, Decision, ProposedAction, RiskClass, now


def payload_hash(arguments: dict[str, Any]) -> str:
    """A stable hash of the canonical JSON of arguments. A changed payload needs a new token."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def summary(action: ProposedAction) -> str:
    """What the user is asked to approve."""
    shown = json.dumps(action.arguments, default=str)
    if len(shown) > 300:
        shown = shown[:300] + "..."
    if action.risk is RiskClass.DESTRUCTIVE:
        effect = "This cannot be undone."
    elif action.risk.at_least(RiskClass.EXTERNAL_WRITE):
        effect = "This acts outside the harness."
    else:
        effect = "This changes state."
    return f"Run {action.tool} with {shown}. {effect}"


class RiskPolicy:
    """The configured policy. Satisfies Policy."""

    def __init__(self, cfg: Policy) -> None:
        self.cfg = cfg

    def authorize(self, action: ProposedAction) -> Decision:
        if action.risk is RiskClass.FORBIDDEN:
            return Decision.deny(f"{action.tool} is forbidden")
        if action.risk is RiskClass.READ_AUTHENTICATED and not self.cfg.allow_authenticated_reads:
            return Decision.deny(
                "authenticated reads are off (agent.policy.allow_authenticated_reads)"
            )
        if action.risk.at_least(self.cfg.confirm_from):
            return Decision.confirm(
                ConfirmationRequest(
                    token=uuid4().hex,
                    tool=action.tool,
                    arguments=action.arguments,
                    payload_hash=payload_hash(action.arguments),
                    risk=action.risk,
                    summary=summary(action),
                    expires_at=now() + timedelta(seconds=self.cfg.confirmation_ttl_secs),
                )
            )
        return Decision.allow()
