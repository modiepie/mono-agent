"""CI smoke eval: the whole graph on the smoke questions, replayed from recorded model and
retrieval responses. No Ollama, no Qdrant, no network. A failure means either the code changed
behaviour or a prompt changed (then re-record: `python -m eval.run --config local-qwen
--subset smoke --record`)."""

import json
from pathlib import Path

from eval import dataset
from eval.recording import Recording
from eval.runner import EvalConfig, run_config

ROOT = Path(__file__).parent.parent / "eval"
RECORDING = ROOT / "recordings" / "local-qwen-smoke.json"
BASELINE = ROOT / "results" / "local-qwen-smoke.json"


def test_smoke_replay_reproduces_recorded_results():
    assert RECORDING.exists(), "recording missing: run the smoke eval with --record"
    questions = dataset.load("smoke")
    results = run_config(
        EvalConfig.load("local-qwen"),
        questions,
        Recording(RECORDING, replay=True),
        log=lambda *_: None,
    )
    assert [r["error"] for r in results] == [None] * len(questions)

    baseline = {q["id"]: q for q in json.loads(BASELINE.read_text("utf-8"))["questions"]}
    for r in results:
        b = baseline[r["id"]]
        assert (r["pass"], r["routing_ok"], r["numbers"], r["tool_args"]) == (
            b["pass"],
            b["routing_ok"],
            b["numbers"],
            b["tool_args"],
        ), r["id"]
        assert r["answer"] == b["answer"], r["id"]
