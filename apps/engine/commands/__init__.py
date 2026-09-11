"""The engine command.

  engine chat / run / confirm / sessions    the agent
  engine skills ...                         the skill bank: list, train, collect, propose
  engine matrix / runs / config             research and inspection

Commands that need the model stack import it lazily, so everything else works without torch.
"""

from engine.commands import agent, chat, research, skills
from engine.commands.app import app

__all__ = ["agent", "app", "chat", "research", "skills"]
