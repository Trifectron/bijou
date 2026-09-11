"""Cases, suites, the runner over a fake engine command, reports and the baseline gate."""

import json
import sys
from pathlib import Path

import pytest

from evals import report as reports
from evals import suites
from evals.core.types import EvalCase, EvalsError, Expect, RunView
from evals.runner import load_cases, run_case

CASES = Path(__file__).resolve().parents[1] / "cases"


def run_view(**overrides):
    data = {
        "session_id": "s1",
        "status": "answered",
        "answer": "Ana works in Tempe",
        "plan": {"fallback": False, "steps": [{"id": "1"}, {"id": "2"}]},
        "steps": [
            {"id": "1", "status": "answered", "skills": ["json_extract"], "tools": ["run_skill"]},
            {"id": "2", "status": "answered", "skills": [], "tools": ["browser_navigate"]},
        ],
        "duration_ms": 1200,
    }
    return RunView.model_validate(data | overrides)


def case(suites_, **expect):
    return EvalCase(id="c", request="r", suites=suites_, expect=Expect(**expect))


def test_the_committed_cases_load_and_name_real_suites():
    loaded = load_cases(CASES)
    assert loaded
    for c in loaded:
        assert set(c.suites) <= set(suites.NAMES), c.id


def test_duplicate_ids_and_bad_lines_are_rejected(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"id": "x", "request": "r", "suites": []}\n' * 2)
    with pytest.raises(EvalsError, match="twice"):
        load_cases(tmp_path)
    (tmp_path / "a.jsonl").write_text('{"id": "x"}\n')
    with pytest.raises(EvalsError, match="a.jsonl:1"):
        load_cases(tmp_path)
    with pytest.raises(EvalsError, match="no cases"):
        load_cases(tmp_path / "empty")


@pytest.mark.parametrize(
    ("name", "expect", "passed"),
    [
        ("status", {"status": "answered"}, True),
        ("status", {"status": "stalled"}, False),
        ("plan", {"min_steps": 2, "max_steps": 2}, True),
        ("plan", {"max_steps": 1}, False),
        ("skills", {"skills": ["json_extract"]}, True),
        ("skills", {"no_skills": True}, False),
        ("tools", {"tools": ["browser_*"]}, True),
        ("tools", {"tools": ["fetch_url"]}, False),
        ("tools", {"no_tools": True}, False),
        ("answer", {"answer_contains": ["tempe", "ANA"]}, True),
        ("answer", {"answer_contains": ["Seattle"]}, False),
        ("latency", {"max_latency_ms": 2000}, True),
        ("latency", {"max_latency_ms": 100}, False),
    ],
)
def test_each_suite_scores_its_expectation(name, expect, passed):
    assert suites.load(name).score(case([name], **expect), run_view()).passed is passed


def test_a_suite_with_nothing_to_check_does_not_count():
    for name in suites.NAMES:
        assert suites.load(name).score(case([name]), run_view()) is None


def test_a_fallback_plan_fails_the_plan_suite():
    view = run_view(plan={"fallback": True, "steps": [{"id": "1"}]})
    assert not suites.load("plan").score(case(["plan"], max_steps=1), view).passed


def fake_engine(tmp_path, body):
    script = tmp_path / "engine.py"
    script.write_text("import json, sys\n" + body)
    return [sys.executable, str(script)]


def test_the_runner_passes_the_request_and_parses_the_result(tmp_path):
    result = run_view().model_dump() | {"request": "r", "context": {"extra": True}}
    command = fake_engine(
        tmp_path,
        f"json.dump(sys.argv[1:], open({str(tmp_path / 'argv.json')!r}, 'w'))\n"
        f"print(json.dumps({result!r}, indent=2))\n",
    )
    assert run_case(case(["status"]), command, 10.0).session_id == "s1"
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv == ["run", "--json", "--user", "eval-c", "r"]


def test_the_runner_reports_a_failed_engine(tmp_path):
    failing = fake_engine(tmp_path, "sys.stderr.write('model offline')\nsys.exit(2)\n")
    with pytest.raises(EvalsError, match="exited 2: model offline.*just agent"):
        run_case(case(["status"]), failing, 10.0)
    garbled = fake_engine(tmp_path, "print('not json')\n")
    with pytest.raises(EvalsError, match="printed no RunResult"):
        run_case(case(["status"]), garbled, 10.0)
    slow = fake_engine(tmp_path, "import time\ntime.sleep(5)\n")
    with pytest.raises(EvalsError, match="no result in 0.5s"):
        run_case(case(["status"]), slow, 0.5)
    with pytest.raises(EvalsError, match="not found"):
        run_case(case(["status"]), [str(tmp_path / "no-such-engine")], 10.0)


def test_reports_aggregate_by_suite_and_compare_to_the_baseline():
    results = reports.score(
        case(["status", "tools"], status="answered", tools=["x"]), run_view(), set(suites.NAMES)
    )
    report = reports.build(results, ["engine"], 1)
    rates = {s.suite: s.rate for s in report.suites}
    assert rates == {"status": 1.0, "tools": 0.0}
    baseline = reports.baseline_of(report)
    assert reports.compare(report, baseline, 0.0) == []
    better = {"status": {"passed": 1, "total": 1}, "tools": {"passed": 1, "total": 1}}
    assert reports.compare(report, better, 0.0) == ["tools: 100% -> 0%"]
    assert reports.compare(report, better, 1.0) == []
    gone = better | {"latency": {"passed": 1, "total": 1}}
    assert "latency: in the baseline, missing from this report" in reports.compare(
        report, gone, 1.0
    )


def test_suites_the_run_did_not_ask_for_are_skipped():
    results = reports.score(
        case(["status", "tools"], status="answered", tools=["x"]), run_view(), {"status"}
    )
    assert [r.suite for r in results] == ["status"]


def test_missing_reports_and_baselines_say_how_to_make_them(tmp_path):
    with pytest.raises(EvalsError, match="evals run"):
        reports.read_report(tmp_path / "r.json")
    with pytest.raises(EvalsError, match="evals baseline"):
        reports.read_baseline(tmp_path / "b.json")
