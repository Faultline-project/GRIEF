"""Trace-first terminal workspace. The runner remains a separate process."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Footer, DataTable, RichLog, TabbedContent, TabPane

from ..cli import new_run_dir, run_options
from ..demo import demo_events
from ..events import EventReader


class TraceView(Static):
    def show_trace(self, trace, requests):
        events = trace.get("events", [])
        observed = {r["request_id"]: r for r in requests}
        duration = max([e.get("offset_ms", 0) for e in events] +
                       [r.get("end_ms", 0) for r in requests] + [1])
        width = 28
        table = Table(expand=True, box=None, padding=(0, 1))
        for title in ("REQUEST", "LIFETIME / SCHEDULE", "EVENT", "CONTEXT"):
            table.add_column(title, style="#b8c8db")
        for e in events[:32]:
            rid, action = e.get("request_id", ""), getattr(e["action"], "value", e["action"])
            start = min(width - 1, max(0, int(e.get("offset_ms", 0) / duration * (width - 1))))
            obs = observed.get(rid) if action == "send" else None
            end = min(width, max(start + 1, int(obs["end_ms"] / duration * width))) if obs else start + 1
            bar = Text(" " * start)
            bar.append(("━" * (end - start)) if obs else "◆", style="#55d6be" if action == "send" else "#ffbe6f")
            bar.append(" " * (width - end))
            context = f"{e.get('prompt_len') or 0} in · {e.get('max_tokens') or 0} out"
            if e.get("prefix_len"):
                context += f" · prefix {e['prefix_len']}"
            if e.get("response_format"):
                context += " · JSON"
            table.add_row(Text(rid), bar, Text(f"{action} @{e.get('offset_ms', 0)}ms"),
                          Text(context if action == "send" else "lifecycle"))
        if len(events) > 32:
            table.add_row("…", "", "", f"{len(events) - 32} more events in saved trace")
        self.update(table)


class GriefApp(App):
    TITLE = "GRIEF"
    CSS_PATH = "app.tcss"
    BINDINGS = [("q", "quit", "Quit"), ("s", "stop", "Stop run"),
                ("r", "replay", "Replay finding"), ("d", "restart_demo", "Restart demo"),
                ("1", "tab('explore')", "Explore"), ("2", "tab('evidence')", "Evidence"),
                ("3", "tab('journal')", "Journal")]

    def __init__(self, *, demo=False, run_dir=None, runner_args=None, autoplay=True, interval=3.0):
        super().__init__()
        self.demo = demo
        self.run_dir = Path(run_dir).resolve() if run_dir else None
        self.runner_args = runner_args
        self.autoplay, self.interval = autoplay, interval
        self.process = None
        self.replay_process = None
        self.replay_busy = False
        self.reader = EventReader(self.run_dir / "events.jsonl") if self.run_dir else None
        self.demo_index = 0
        self.trace = {}
        self.requests = []
        self.findings = []
        self.selected = None
        self.candidates = 0
        self.iteration = 0
        self.corpus = 0
        self.mode = "SIMULATED · no server" if demo else ("LIVE" if runner_args is not None else "RECORDED / WATCH")

    def compose(self) -> ComposeResult:
        with Horizontal(id="masthead"):
            yield Static("GRIEF", id="brand")
            yield Static("CONCURRENT SERVING EXPLORER", id="tagline")
            yield Static(self.mode, id="mode")
        yield Static("Waiting for a run", id="target", markup=False)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("SESSION", classes="eyebrow")
                yield Static("Ready", id="phase", markup=False)
                yield Static("0\niterations", id="iterations", classes="metric")
                yield Static("0\ncorpus traces", id="corpus", classes="metric")
                yield Static("0\ncandidate observations", id="candidates", classes="metric")
                yield Static("0\nsignature exemplars", id="signatures", classes="metric")
                yield Static("FEEDBACK", classes="eyebrow")
                yield Static("No observations yet", id="feedback", markup=False)
            with TabbedContent(id="tabs"):
                with TabPane("Explore", id="explore"):
                    yield Static("REQUEST INTERACTIONS", classes="eyebrow")
                    yield Static("Select → mutate → execute → evaluate → retain", id="pipeline")
                    yield Static("Waiting for a trace", id="mutations", markup=False)
                    yield TraceView(id="trace")
                    yield Static("◆ scheduled event   ━ observed request lifetime", id="legend")
                    yield Static("FINDINGS", classes="eyebrow")
                    yield DataTable(id="findings", cursor_type="row")
                with TabPane("Evidence", id="evidence"):
                    with VerticalScroll():
                        yield Static("Select a finding in Explore to inspect its evidence.", id="detail", markup=False)
                with TabPane("Journal", id="journal"):
                    yield RichLog(id="journal_log", wrap=True, highlight=False, markup=False, max_lines=500)
        yield Static("Timed workloads · serving-state feedback · controlled confirmation", id="narration", markup=False)
        yield Footer()

    def on_mount(self):
        self.query_one("#findings", DataTable).add_columns("SIGNATURE", "ORACLE", "REDUCED TRACE")
        self.query_one("#trace", TraceView).show_trace({}, [])
        if self.demo:
            if self.autoplay:
                self.set_interval(self.interval, self.advance_demo)
            self.advance_demo()
        else:
            self.set_interval(0.2, self.poll_events)
            if self.runner_args is not None:
                self.start_runner()

    def on_resize(self, event):
        self.screen.set_class(event.size.width < 105 or event.size.height < 35, "compact")

    def advance_demo(self):
        events = demo_events()
        if self.demo_index < len(events):
            self.consume(events[self.demo_index])
            self.demo_index += 1

    def poll_events(self):
        if self.reader:
            for record in self.reader.read():
                self.consume(record)

    def consume(self, record):
        kind, data = record["type"], record["data"]
        phase = self.query_one("#phase", Static)
        self.query_one("#journal_log", RichLog).write(Text(
            f"{record.get('elapsed_s', 0):7.1f}s  {kind}  " +
            str(data.get("verdict", data.get("message", data.get("text", ""))))))
        if kind == "run_started":
            config = data.get("config", {})
            self.query_one("#target", Static).update(f"{config.get('model_name') or 'Auto-discover model'}   /   {config.get('base_url', '')}")
            phase.update("Preparing")
        elif kind == "trace_started":
            self.trace, self.requests = data["trace"], []
            self.iteration = data.get("iteration", 0)
            self.corpus = data.get("corpus", self.corpus)
            phase.update("Executing trace")
            self.query_one("#mutations", Static).update(" + ".join(data.get("mutations", [])) or "Seed trace")
        elif kind == "iteration_finished":
            self.iteration = data.get("iteration", self.iteration) + 1
            self.corpus = data.get("corpus", self.corpus)
            self.candidates += data.get("candidates", 0)
            self.requests = data.get("requests", [])
            phase.update("Evaluating")
            gaps = data.get("measurement_gaps", [])
            self.query_one("#feedback", Static).update(
                ("New state retained\n" if data.get("novel") else "State revisited\n") +
                ("Unavailable checks: " + "; ".join(map(str, gaps)) if gaps else str(data.get("fingerprint", "Unavailable"))))
        elif kind == "finding_saved":
            self.findings.append(data)
            finding = data["finding"]
            self.query_one("#findings", DataTable).add_row(
                Text(str(finding.get("mech_sig", "unknown"))[:14]),
                Text(", ".join(b["class"] for b in finding.get("bugs", []))),
                Text(finding.get("minimized_validation", {}).get("status", "unknown")),
                key=str(len(self.findings) - 1))
            if self.selected is None:
                self.select_finding(0)
            phase.update("Finding saved")
        elif kind == "confirmation":
            phase.update("Confirmation recorded")
            self.query_one("#narration", Static).update(str(data.get("verdict", "")) + " · " + str(data.get("reason", "See confirmation_log.jsonl for replay evidence")))
        elif kind == "minimization_started":
            phase.update("Reducing trace")
        elif kind == "minimization_finished":
            phase.update("Reduced trace checked")
        elif kind == "run_finished":
            phase.update(data.get("status", "finished").capitalize())
        elif kind == "error":
            phase.update("Failed")
            self.query_one("#narration", Static).update(str(data.get("message", "Unknown error")))
        elif kind == "narration":
            self.query_one("#narration", Static).update(data["text"])
        for key, value, label in (("iterations", self.iteration, "iterations"),
                                  ("corpus", self.corpus, "corpus traces"),
                                  ("candidates", self.candidates, "candidate observations"),
                                  ("signatures", len({f['finding'].get('mech_sig') for f in self.findings}), "signatures")):
            self.query_one("#" + key, Static).update(f"{value}\n{label}")
        self.query_one("#trace", TraceView).show_trace(self.trace, self.requests)

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        self.select_finding(int(event.row_key.value))
        self.action_tab("evidence")

    def select_finding(self, index):
        self.selected = index
        data = self.findings[index]
        f = data["finding"]
        text = ("SIMULATED TEACHING EXAMPLE\n\n" if f.get("demo") else "SAVED FINDING\n\n")
        text += "Signature: " + str(f.get("mech_sig")) + "\n"
        text += "Reduced trace: " + f.get("minimized_validation", {}).get("status", "unknown") + "\n"
        text += "\n" + "\n\n".join(b.get("detail", "") for b in f.get("bugs", []))
        text += "\n\nEVIDENCE\n" + json.dumps(f.get("bugs", []), indent=2)
        text += "\n\nSTORED TRACE\n" + json.dumps(f.get("trace", {}), indent=2)
        if f.get("trace_original"):
            text += "\n\nORIGINAL TRACE\n" + json.dumps(f["trace_original"], indent=2)
        self.query_one("#detail", Static).update(text)

    def action_tab(self, tab):
        self.query_one("#tabs", TabbedContent).active = tab

    def action_restart_demo(self):
        if self.demo:
            self.demo_index = self.candidates = self.iteration = self.corpus = 0
            self.findings.clear()
            self.selected = None
            self.trace, self.requests = {}, []
            self.query_one("#findings", DataTable).clear()
            self.query_one("#detail", Static).update("Select a finding to inspect its evidence.")
            self.advance_demo()

    @work(exclusive=True, exit_on_error=False)
    async def start_runner(self):
        log_path = self.run_dir / "runner.log"
        with log_path.open("w") as log:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "fuzzer", *run_options(self.runner_args, self.run_dir),
                stdout=log, stderr=asyncio.subprocess.STDOUT)
            code = await self.process.wait()
        self.poll_events()
        if code:
            self.query_one("#phase", Static).update(f"Exited ({code})")
            self.query_one("#narration", Static).update(log_path.read_text()[-1000:])

    async def stop_process(self, process):
        if process is not None and process.returncode is None:
            try:
                process.send_signal(signal.SIGINT)
                await asyncio.wait_for(process.wait(), timeout=15)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                self.notify("Worker exceeded shutdown timeout; existing files were preserved.", severity="warning")
            except ProcessLookupError:
                pass

    async def action_stop(self):
        await self.stop_process(self.process)
        await self.stop_process(self.replay_process)
        self.poll_events()

    async def action_quit(self):
        await self.action_stop()
        self.exit()

    @work(exit_on_error=False)
    async def action_replay(self):
        if self.replay_busy:
            self.notify("Replay is already running.")
            return
        if self.selected is None:
            self.notify("Select a finding first.")
            return
        if self.demo or self.findings[self.selected]["finding"].get("demo"):
            self.notify("This is a simulated example. Live findings can be replayed against your server.")
            return
        if self.runner_args is None:
            self.notify("Use grief replay with the target server and adapter flags for this recording.")
            return
        if self.process is not None and self.process.returncode is None:
            self.notify("Stop the fuzzing run before controlled replay.")
            return
        output = self.run_dir / f"replay-{len(list(self.run_dir.glob('replay-*.json'))):03d}.json"
        self.replay_busy = True
        try:
            with (self.run_dir / "replay.log").open("w") as log:
                self.replay_process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "fuzzer.cli", "replay", self.findings[self.selected]["path"],
                    "--output", str(output), *self.runner_args,
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
                code = await self.replay_process.wait()
        finally:
            self.replay_busy = False
        self.query_one("#detail", Static).update(output.read_text() if code == 0 and output.exists()
            else (self.run_dir / "replay.log").read_text())
        self.action_tab("evidence")


def launch(argv, demo=False):
    p = argparse.ArgumentParser(prog="grief demo" if demo else "grief tui",
        description="Explore timed requests and serving-state feedback. Server/adapter flags match grief run.")
    p.add_argument("--watch", type=Path, help="Read events from an existing run without starting a worker")
    p.add_argument("--speed", type=float, default=1.0, help="Demo playback speed")
    args, rest = p.parse_known_args(argv)
    if args.speed <= 0:
        p.error("--speed must be positive")
    if demo and rest:
        p.error("demo is offline; it does not take server flags")
    if args.watch and not (args.watch / "events.jsonl").is_file():
        p.error("watch directory must contain events.jsonl")
    from ..__main__ import parse_args
    if not demo and not args.watch:
        parse_args(rest)
    run_dir = args.watch if args.watch else (None if demo else new_run_dir())
    GriefApp(demo=demo, run_dir=run_dir, runner_args=None if demo or args.watch else rest,
             interval=3.0 / args.speed).run()
