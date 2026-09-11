"""The session index and the pattern miner that reads it."""

import json
from datetime import UTC, datetime, timedelta

from engine.core.config import Patterns
from engine.core.types.agent import Plan, PlanStep, RunStatus, SessionRecord, StepReport
from engine.memory.patterns import PatternMiner, keywords, write_proposals
from engine.memory.sessions import SqliteSessionStore, fts_query

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def record(i, request, goals, skills=(), status=RunStatus.ANSWERED):
    steps = [
        StepReport(id=str(n), goal=g, status=status, answer=f"answer to {g}", skills=list(skills))
        for n, g in enumerate(goals)
    ]
    return SessionRecord(
        id=f"s{i}",
        created_at=T0 + timedelta(minutes=i),
        updated_at=T0 + timedelta(minutes=i),
        request=request,
        status=status,
        answer=f"answer {i}",
        plan=Plan(steps=[PlanStep(id=s.id, goal=s.goal) for s in steps]),
        steps=steps,
    )


def test_records_round_trip_and_list_newest_first(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    store.save(record(1, "first", ["a"]))
    store.save(record(2, "second", ["b"]))
    assert store.get("s1") == record(1, "first", ["a"])
    assert store.get("missing") is None
    assert [s.id for s in store.recent(10)] == ["s2", "s1"]
    store.save(record(1, "first again", ["a"]))
    assert store.get("s1").request == "first again"
    assert len(store.recent(10)) == 2


def test_search_covers_requests_answers_and_step_goals(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    store.save(record(1, "amazon scientist jobs", ["search the careers page"]))
    store.save(record(2, "weather in tempe", ["fetch forecast"]))
    assert [s.id for s in store.search("Amazon", 5)] == ["s1"]
    assert [s.id for s in store.search("careers", 5)] == ["s1"]
    assert [s.id for s in store.search("forecasts", 5)] == ["s2"]
    assert store.search('" OR 1=1 --', 5) == []
    assert store.search("", 5) == []


def test_user_text_cannot_become_fts_syntax():
    assert fts_query('jobs" OR NEAR(x') == '"jobs" "or" "near" "x"'


def test_keywords_drop_stopwords_and_short_words():
    assert keywords("Find the latest Amazon scientist jobs in AZ") == {
        "amazon",
        "scientist",
        "jobs",
    }


def mined(store, **overrides):
    return PatternMiner(store, Patterns(**({"min_occurrences": 3} | overrides)))


def test_recurring_uncovered_work_becomes_one_proposal(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    for i, goal in enumerate(
        [
            "summarize amazon scientist job postings",
            "summarize new amazon scientist job listings",
            "summarize amazon applied scientist job postings",
        ]
    ):
        store.save(record(i, "jobs", [goal]))
    store.save(record(9, "weather", ["fetch the tempe forecast"]))
    store.save(
        record(10, "bio", ["summarize amazon scientist job postings"], skills=["json_extract"])
    )
    proposals = mined(store).propose()
    assert len(proposals) == 1
    p = proposals[0]
    assert p.sessions == ["s0", "s1", "s2"]
    assert set(p.name.split("_")) <= {"summarize", "amazon", "scientist", "job", "postings"}
    assert len(p.pairs) == 3 and p.pairs[0].output.startswith("answer to")
    assert p.approved is False


def test_one_session_repeating_itself_is_not_a_pattern(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    store.save(record(1, "x", ["draft cover letter"] * 5))
    assert mined(store).propose() == []


def test_names_avoid_existing_skills_and_files_are_never_overwritten(tmp_path):
    store = SqliteSessionStore(tmp_path / "s.db")
    for i in range(3):
        store.save(record(i, "x", ["draft cover letter"]))
    first = mined(store).propose()[0]
    renamed = mined(store).propose(existing={first.name})[0]
    assert renamed.name == f"{first.name}_2"
    written, skipped = write_proposals([first], tmp_path / "proposals")
    assert json.loads(written[0].read_text())["approved"] is False
    assert write_proposals([first], tmp_path / "proposals") == ([], written)
