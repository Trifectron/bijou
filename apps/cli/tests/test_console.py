"""The developer console: catalog, command line, logs, status parsing, runner, and the app."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

import pytest

from cli.core.config import Config
from cli.logs import LogBuffer, LogLine, LogWriter, log_name
from cli.runner import Runner
from cli.status import Snapshot, last_run, parse_gpus, parse_holders
from cli.units import Command, Group, Unit, catalog, parse_command

ROOT = Path(__file__).resolve().parents[3]
KNOWN = ("json_extract",)


@pytest.fixture
def cfg() -> Config:
    return Config()


def _recipes() -> set[str]:
    text = (ROOT / "justfile").read_text()
    names = set(re.findall(r"^([a-z][\w-]*)(?:\s[^:\n]*)?:(?!=)", text, re.MULTILINE))
    return names | set(re.findall(r"^alias ([\w-]+) :=", text, re.MULTILINE))


def test_every_catalog_unit_is_a_just_recipe():
    recipes = _recipes()
    for unit in catalog(KNOWN):
        assert unit.args[0] in recipes, unit.id


def test_every_skill_has_a_train_unit():
    ids = {u.id for u in catalog(("a", "b"))}
    assert {"skills train a", "skills train b"} <= ids


def test_catalog_ids_are_unique():
    ids = [u.id for u in catalog(KNOWN)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("q", Command("quit")),
        ("help", Command("help")),
        ("clear", Command("clear")),
        ("start check", Command("start", arg="check")),
        ("stop skill train json_extract", Command("stop", arg="skill train json_extract")),
        ("restart check", Command("restart", arg="check")),
        ("just runs", Command("just", args=("runs",))),
        (
            "skill sample json_extract -n 2",
            Command("just", args=("skill", "sample", "json_extract", "-n", "2")),
        ),
        ("start", Command("just", args=("start",))),
        ("", Command("none")),
    ],
)
def test_command_line_parsing(text, expected):
    assert parse_command(text) == expected


def _line(text: str) -> LogLine:
    return LogLine(datetime(2026, 1, 1), "out", text)


def test_the_log_buffer_keeps_the_newest_lines():
    buffer = LogBuffer(3)
    for i in range(5):
        buffer.append(_line(f"line {i}"))
    assert [line.text for line in buffer.window(0, 10)] == ["line 2", "line 3", "line 4"]


def test_search_wraps_in_either_direction():
    buffer = LogBuffer(10)
    for text in ("alpha", "Beta", "gamma", "beta"):
        buffer.append(_line(text))
    assert buffer.find("beta", 0) == 1
    assert buffer.find("beta", 2) == 3
    assert buffer.find("beta", 0, backwards=True) == 3
    assert buffer.find("delta", 0) is None


def test_log_files_are_named_by_unit_and_appended(tmp_path):
    writer = LogWriter(tmp_path)
    writer.append("skill train json_extract --full-finetune", _line("hello"))
    writer.close()
    name = log_name("skill train json_extract --full-finetune")
    assert name == "skill_train_json_extract_--full-finetune.log"
    assert (tmp_path / name).read_text().endswith("out hello\n")


def test_nvidia_smi_output_parses():
    gpus = parse_gpus("NVIDIA GeForce RTX 4050 Laptop GPU, 5479, 6141\n")
    assert gpus[0].used_mib == 5479 and gpus[0].total_mib == 6141
    assert parse_holders("852888, /app/llama-server\n853038, /app/llama-server\n") == (
        "llama-server",
        "llama-server",
    )
    assert parse_gpus("") == []


def test_the_newest_run_record_is_the_last_run(tmp_path):
    for name, kind, started in (
        ("a", "train", "2026-09-01T10:00:00"),
        ("b", "collect", "2026-09-02T11:30:00"),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "record.json").write_text(
            f'{{"kind": "{kind}", "started_at": "{started}"}}'
        )
    assert last_run(tmp_path) == "collect 09-02 11:30"
    assert last_run(tmp_path / "missing") == ""


def test_the_console_reads_only_its_keys_from_the_shared_file(tmp_path, monkeypatch):
    path = tmp_path / "bijou.toml"
    path.write_text("[console]\nlog_lines = 7\n[agent.http]\nport = 9999\n[train]\nlr = 0.1\n")
    monkeypatch.setenv("BIJOU_CONFIG_FILE", str(path))
    cfg = Config()
    assert cfg.console.log_lines == 7
    assert cfg.agent.http.port == 9999


def _collect(tmp_path, script: str, stop_after: float | None = None):
    lines: list[tuple[str, str]] = []
    codes: list[int | None] = []

    async def scenario() -> None:
        done = asyncio.Event()

        def on_exit(_unit: str, code: int | None) -> None:
            codes.append(code)
            done.set()

        runner = Runner(tmp_path, lambda _u, s, t: lines.append((s, t)), on_exit, ("sh", "-c"))
        await runner.start("u", [script])
        assert runner.owns("u")
        if stop_after is not None:
            await asyncio.sleep(stop_after)
            assert runner.stop("u")
        await asyncio.wait_for(done.wait(), timeout=10)
        assert not runner.owns("u")

    asyncio.run(scenario())
    return lines, codes


def test_the_runner_streams_both_outputs_and_the_exit_code(tmp_path):
    lines, codes = _collect(tmp_path, "echo hello; echo oops >&2; exit 3")
    assert ("out", "hello") in lines
    assert ("err", "oops") in lines
    assert lines[0][0] == "meta"
    assert codes == [3]


def test_stopping_kills_the_whole_process_group(tmp_path):
    lines, codes = _collect(tmp_path, "sleep 30 & sleep 30; echo never", stop_after=0.3)
    assert codes and codes[0] != 0
    assert ("out", "never") not in lines


# ---------- the app, driven headless ----------


def _snapshot(_cfg) -> Snapshot:
    return Snapshot(
        gpus=(),
        checkpoint="",
        checkpoint_present=False,
        adapters=0,
        full_finetunes=0,
        skills=1,
        last_run="",
        git="abc1234",
    )


def _app(cfg, tmp_path, *scripts: str):
    from cli.app import ConsoleApp

    units = [Unit(Group.GATE, (s,), f"hint {i}") for i, s in enumerate(scripts)]
    return ConsoleApp(cfg, tmp_path, units=units, launcher=("sh", "-c"), probe=_snapshot)


async def _until(pilot, condition, timeout: float = 10.0) -> None:
    for _ in range(int(timeout / 0.05)):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition not met in time")


def test_enter_runs_the_selected_unit_and_streams_its_output(cfg, tmp_path):
    pytest.importorskip("textual")
    from cli.app import Status

    async def scenario() -> None:
        app = _app(cfg, tmp_path, "echo first", "echo second")
        async with app.run_test(size=(100, 30)) as pilot:
            assert app.query_one("#units").size.height > 20
            assert app.query_one("#logs").size.height > 20
            await pilot.press("j", "enter")
            await _until(pilot, lambda: app.states[1].status is Status.OK)
            texts = [line.text for line in app.states[1].logs.window(0, 10)]
            assert "second" in texts
            assert texts[-1] == "exited with code 0"
            assert app.states[0].status is Status.IDLE
        assert (tmp_path / cfg.console.log_dir / log_name("echo second")).exists()

    asyncio.run(scenario())


def test_x_stops_a_running_unit(cfg, tmp_path):
    pytest.importorskip("textual")
    from cli.app import Status

    async def scenario() -> None:
        app = _app(cfg, tmp_path, "sleep 30")
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await _until(pilot, lambda: app.runner.owns("sleep 30"))
            await pilot.press("x")
            await _until(pilot, lambda: not app.runner.owns("sleep 30"))
            assert app.states[0].status is Status.IDLE
            assert app.states[0].logs.window(0, 10)[-1].text == "stopped"

    asyncio.run(scenario())


def test_a_failing_unit_is_marked_failed(cfg, tmp_path):
    pytest.importorskip("textual")
    from cli.app import Status

    async def scenario() -> None:
        app = _app(cfg, tmp_path, "exit 2")
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await _until(pilot, lambda: app.states[0].status is Status.FAILED)
            assert app.states[0].code == 2

    asyncio.run(scenario())


def test_the_command_line_and_search(cfg, tmp_path):
    pytest.importorskip("textual")
    from cli.app import Mode, Pane, Status

    async def scenario() -> None:
        # The word is assembled at run time, so only the output line contains it.
        app = _app(cfg, tmp_path, "printf 'a\\nne%sle\\nb\\n' ed")
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await _until(pilot, lambda: app.states[0].status is Status.OK)

            await pilot.press("slash", "n", "e", "e", "d", "l", "e", "enter")
            assert app.key_mode is Mode.NORMAL
            assert app.pane is Pane.LOGS
            assert app.hit is not None
            assert app.states[0].logs.window(app.hit, 1)[0].text == "needle"

            await pilot.press("colon", "c", "l", "e", "a", "r", "enter")
            assert len(app.states[0].logs) == 0

            await pilot.press("tab")
            assert app.pane is Pane.UNITS
            await pilot.press("question_mark")
            assert app.show_help
            await pilot.press("j")
            assert not app.show_help
            assert app.selected == 0

    asyncio.run(scenario())


def test_an_unknown_command_runs_as_an_adhoc_recipe(cfg, tmp_path):
    pytest.importorskip("textual")

    async def scenario() -> None:
        app = _app(cfg, tmp_path, "true")
        async with app.run_test() as pilot:
            await pilot.press("colon", *"echo", "enter")
            assert app.states[-1].unit.group is Group.ADHOC
            assert app.states[-1].unit.id == "echo"
            assert app.selected == len(app.states) - 1

    asyncio.run(scenario())
