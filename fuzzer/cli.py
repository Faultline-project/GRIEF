"""Public commands; UI imports stay optional for headless installations."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .events import EventWriter, json_default


def new_run_dir(root="runs"):
    name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    path = Path(root) / name
    path.mkdir(parents=True)
    return path.resolve()


def run_options(argv, run_dir):
    values = {"--findings-dir": str(run_dir / "findings"),
              "--corpus-dir": str(run_dir / "corpus"),
              "--events-file": str(run_dir / "events.jsonl")}
    result = list(argv)
    for option, value in values.items():
        if not any(a == option or a.startswith(option + "=") for a in argv):
            result += [option, value]
    return result


async def run_headless(argv):
    from .__main__ import main as fuzz
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except NotImplementedError:
            pass
    await fuzz(argv)


def inspect_run(path):
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text())
    findings = path / "findings" if (path / "findings").is_dir() else path
    result = {"run": str(path.resolve()), "findings": []}
    for file in sorted(findings.glob("finding_*.json")):
        report = json.loads(file.read_text())
        result["findings"].append({"path": str(file), "signature": report.get("mech_sig"),
            "classes": [b.get("class") for b in report.get("bugs", [])],
            "minimized_validation": report.get("minimized_validation", {"status": "unknown"})})
    for name in ("signature_ledger", "confirmation_summary", "run_meta"):
        file = findings / (name + ".json")
        if file.exists():
            result[name] = json.loads(file.read_text())
    if not result["findings"] and not path.exists():
        raise ValueError(f"run does not exist: {path}")
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="grief", description="Explore concurrent LLM serving behavior.")
    parser.add_argument("--version", action="version", version="GRIEF 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", add_help=False, help="Fuzz a live inference server")
    sub.add_parser("tui", add_help=False, help="Explore a live or recorded run")
    sub.add_parser("demo", add_help=False, help="Offline simulated walkthrough (no server)")
    sub.add_parser("replay", add_help=False, help="Replay a finding or trace against a server")
    sub.add_parser("inspect", add_help=False, help="Read findings and run metadata as JSON")
    args, rest = parser.parse_known_args(argv)
    try:
        if args.command in {"tui", "demo"}:
            try:
                from .tui.app import launch
            except ImportError as exc:
                raise ValueError('TUI dependencies missing; install with pip install ".[tui]"') from exc
            launch(rest, demo=args.command == "demo")
        elif args.command == "inspect":
            p = argparse.ArgumentParser(prog="grief inspect")
            p.add_argument("path")
            parsed = p.parse_args(rest)
            print(json.dumps(inspect_run(parsed.path), indent=2, default=json_default))
        elif args.command == "replay":
            from .__main__ import parse_args
            from .replay import replay
            p = argparse.ArgumentParser(prog="grief replay", description="Replay contracts using the same executor and oracles. Pass server/adapter flags as for grief run.")
            p.add_argument("finding")
            p.add_argument("--runs", type=int, default=3)
            p.add_argument("--original", action="store_true", help="Replay trace_original instead of the reduced trace")
            p.add_argument("--output", type=Path)
            parsed, extra = p.parse_known_args(rest)
            if parsed.runs < 1:
                raise ValueError("--runs must be positive")
            runner_args = parse_args(extra)
            if runner_args.confirm_ladder:
                raise ValueError("for controlled confirmation use grief run with --confirm-ladder and the recorded trace as a seed")
            result = asyncio.run(replay(parsed.finding, runner_args, runs=parsed.runs, original=parsed.original))
            serialized = json.dumps(result, indent=2, default=json_default)
            if parsed.output:
                parsed.output.parent.mkdir(parents=True, exist_ok=True)
                parsed.output.write_text(serialized + "\n")
            print(serialized)
        else:
            from .__main__ import parse_args
            # Validate and show help before creating any run directory.
            parsed = parse_args(rest)
            run_dir = Path(parsed.events_file).resolve().parent if parsed.events_file else new_run_dir()
            run_dir.mkdir(parents=True, exist_ok=True)
            options = run_options(rest, run_dir)
            print(f"[grief] run directory: {run_dir}", flush=True)
            try:
                asyncio.run(run_headless(options))
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("[grief] stopped; saved findings remain on disk")
            except Exception as exc:
                EventWriter(parse_args(options).events_file).emit("error", message=str(exc))
                raise
    except (ValueError, OSError) as exc:
        parser.exit(2, f"grief: {exc}\n")


if __name__ == "__main__":
    main()
