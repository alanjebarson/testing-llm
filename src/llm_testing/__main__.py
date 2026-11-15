"""CLI: python -m llm_testing {generate,judge,run} [--config PATH] [--input PATH]"""

from __future__ import annotations

import argparse

from .config import load_settings, project_path
from .io_utils import read_sheet, write_sheet
from .report import print_summary, summarize


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="llm_testing", description=__doc__)
    ap.add_argument("command", choices=["generate", "judge", "run"])
    ap.add_argument("--config", help="path to settings.yaml")
    ap.add_argument("--input", help="override the input sheet for this step")
    args = ap.parse_args(argv)

    settings = load_settings(args.config)
    paths = settings["paths"]

    from .client import make_client  # imported late so --help works without the openai package

    # Build both clients up front so a missing key or a stopped Ollama server is
    # reported before a long generation run, not after it.
    actor_client = judge_client = None
    if args.command in ("generate", "run"):
        actor_client = make_client(settings["actor"], settings["retry"])
        actor_client.check_model(settings["actor"]["model"])
        print(f"Actor: {settings['actor']['provider']} / {settings['actor']['model']}")
    if args.command in ("judge", "run"):
        judge_client = make_client(settings["judge"], settings["retry"])
        print(f"Judge: {settings['judge']['provider']} / {settings['judge']['model']}")

    if args.command in ("generate", "run"):
        from .actor import run_actor

        src = project_path(args.input or paths["test_cases"])
        cases = read_sheet(src)
        print(f"Loaded {len(cases)} test cases from {src}")
        out = write_sheet(run_actor(cases, actor_client, settings), project_path(paths["responses"]))
        print(f"Responses saved: {out}")

    if args.command in ("judge", "run"):
        from .judge import run_judge

        src = project_path(args.input if args.command == "judge" and args.input else paths["responses"])
        responses = read_sheet(src, required=["Test_ID", "Category", "Prompt", "Expected_Behavior", "Response"])
        results = run_judge(responses, judge_client, settings)
        out = write_sheet(results, project_path(paths["results"]))
        print_summary(summarize(results, settings["thresholds"]["fine_tuning_fail_rate"]))
        print(f"Results saved: {out}")


if __name__ == "__main__":
    main()
