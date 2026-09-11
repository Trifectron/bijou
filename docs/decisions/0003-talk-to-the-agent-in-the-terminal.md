# 0003 — Talk to the agent in the terminal, not over HTTP

**Context.** There were three ways to reach the agent and none of them was a conversation:
`engine run` for one request, `POST /run` on `engine serve`, and the console, which shelled out to
`engine run` and showed the output in a log pane. A follow-up meant copying a session id into
`--resume`, and approving a held action meant copying a token into `engine confirm`. The HTTP
surface cost `engine/api` (two apps, about 190 lines), `RemoteBank`, fastapi and uvicorn, and it
was carrying one real user, `apps/evals`, which posts one request per case and could as easily run
the command. `agent.skills.mode = "http"` and the bank server behind it have never been used: the
bank runs in the agent's process on the one machine there is. The console had no chat pane, and
what the agent did between question and answer — its plan, its skill picks, its tool calls — was
visible only in Phoenix or a trace file.

**Decision.** One way in, the terminal. `engine chat` holds a warm agent and takes turns, either as
a prompt for a person or, with `--jsonl`, as one JSON object per line for the console to drive;
each turn continues the session before it until `/new`, and a held action is approved in the same
place it is raised. The console runs that process from the start and puts the conversation in a
third pane, with every trace event streaming into the agent's log beside it. `engine serve`,
`engine serve --bank`, `engine.api`, `RemoteBank`, `[agent.http]`, `[agent.skills].mode` and
`.url`, fastapi and uvicorn are deleted; `[serve]` becomes `[bank]`, holding only the generation
ceiling the in-process bank reads. `apps/evals` runs `engine run --json --user eval-<id>` per case.
One listening port survives: `engine chat` serves the Prometheus registry at
`telemetry.metrics_port`, 9464, and answers nothing else, because a scrape has to reach in and the
dashboard's sixteen agent and skill-bank panels would otherwise go dark.

**Consequences.** The loop, the confirmation flow and the session chain now have a single
implementation, and the console shows a run as it happens rather than after it. Four dependencies
and two servers are gone. Against that: metrics are counted only while a chat process is up, so a
bare `engine run` contributes nothing to Grafana, and Phoenix spans are the only record of it;
evals pay agent startup per case, which is seconds now and will be the skill bank's load time once
adapters exist; and nothing off this machine can reach the agent any more — no second client, no
teammate, no phone. Reversing that means writing a server again, not flipping a setting. The
remote skill bank that `0002` left as an option is gone with it, so a laptop cannot borrow a GPU
box's bank today.

**Not decided.** Whether evals should drive one warm `engine chat --jsonl` process instead of a
process per case, which is the obvious fix if startup ever dominates a run. Whether the bank gets
a server again when training moves to a rented GPU, which `0002` anticipated and this record does
not foreclose. Whether metrics should be pushed rather than scraped, which would remove the last
port. What would change our mind: a second client that must reach this agent from another machine,
or an eval suite slow enough that per-case startup shows up in the numbers.
