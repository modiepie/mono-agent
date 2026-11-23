"""One command for the whole evaluation.

uv run python -m eval.run --config local-qwen          # full run, writes results + report
uv run python -m eval.run --config local-qwen --subset smoke --record
uv run python -m eval.run --config local-qwen --subset smoke --replay   # offline (CI)
uv run python -m eval.run --ablation                   # RAG ablation (no LLM)
uv run python -m eval.run --report                     # rebuild report.md only
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from eval import dataset, report
from eval.recording import Recording
from eval.report import not_run, summarize
from eval.runner import EvalConfig, rescore, run_config, save

RESULTS = Path(__file__).parent / "results"
RECORDINGS = Path(__file__).parent / "recordings"


def main() -> None:
    load_dotenv()  # provider SDKs read API keys from the environment
    p = argparse.ArgumentParser(prog="eval.run")
    p.add_argument("--config", help="name of a file in eval/configs")
    p.add_argument("--subset", choices=["smoke", "core"], default=None)
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--only", nargs="*", default=None, help="question ids")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="keep finished answers from the last run, re-run only rate-limited ones",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="save responses for replay")
    mode.add_argument("--replay", action="store_true", help="use saved responses, no network")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument(
        "--rescore",
        action="store_true",
        help="re-score stored results against the current dataset (no model calls)",
    )
    args = p.parse_args()

    questions = dataset.load(args.subset)
    if args.only and not args.resume:  # with --resume, --only means "redo these, keep the rest"
        questions = [q for q in questions if q.id in args.only]
    if args.limit:
        questions = questions[: args.limit]

    if args.ablation:
        from eval.ablation import run_ablation

        rows = run_ablation(questions)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "ablation.json").write_text(json.dumps(rows, indent=1), "utf-8")

    if args.config:
        cfg = EvalConfig.load(args.config)
        if missing := cfg.missing_keys():
            sys.exit(f"{cfg.name}: set {', '.join(missing)} to run this config")
        recording = None
        if args.record or args.replay:
            recording = Recording(
                RECORDINGS / f"{cfg.name}-{args.subset or 'full'}.json", replay=args.replay
            )
        suffix = f"-{args.subset}" if args.subset else ""
        out = RESULTS / f"{cfg.name}{suffix}.json"
        done: dict[str, dict] = {}
        if args.resume and out.exists():
            previous = json.loads(out.read_text("utf-8"))["questions"]
            redo = set(args.only or [])
            done = {q["id"]: q for q in previous if not not_run(q) and q["id"] not in redo}
            print(f"resuming: {len(done)} kept, {len(questions) - len(done)} to run")
        todo = [q for q in questions if q.id not in done]

        def checkpoint(so_far: list[dict]) -> None:
            # Finished questions only, in dataset order: `--resume` picks up from here.
            new = {r["id"]: r for r in so_far}
            save(
                [done.get(q.id) or new[q.id] for q in questions if q.id in done or q.id in new],
                cfg,
                out,
            )

        fresh = {
            r["id"]: r
            for r in run_config(
                cfg, todo, recording, use_judge=not args.no_judge, on_result=checkpoint
            )
        }
        results = [done.get(q.id) or fresh[q.id] for q in questions]
        if recording:
            recording.save()
        save(results, cfg, out)
        s = summarize(results)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}))

    if args.rescore:
        by_id = {q.id: q for q in dataset.load()}
        for path in [*RESULTS.glob("*.json"), *RESULTS.glob("baseline/*.json")]:
            if path.stem == "ablation":
                continue
            data = json.loads(path.read_text("utf-8"))
            data["questions"] = [rescore(r, by_id[r["id"]]) for r in data["questions"]]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
            print(f"rescored {path.name}")

    if args.config or args.ablation or args.report or args.rescore:
        print(f"wrote {report.write()}")


if __name__ == "__main__":
    main()
