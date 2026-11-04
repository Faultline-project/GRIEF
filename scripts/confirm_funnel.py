#!/usr/bin/env python3
"""Summarise confirmation-ladder runs as a funnel.

Reads every ``confirmation_log.jsonl`` under the given paths (files or
directories, searched recursively) and, when present beside it, the run's
``confirmation_summary.json`` for the candidates that never reached the queue
(signature cooldown).

    python3 scripts/confirm_funnel.py $SCRATCH/fse_ladder/*/findings
    python3 scripts/confirm_funnel.py --json run1/findings run2/findings

Rows: one per run, then the total.  Columns: every verdict the log contains,
split by the logit screen's priority (nontie / tie), so the table answers both
"how much did the ladder filter" and "what happened to the FP ties that used
to be dropped".
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ORDER = ["BI_BATCH_DEPENDENT", "BI_SCHEDULE_DEPENDENT", "BI_RELATION_PERSISTS",
         "SOLO_NONDETERMINISTIC", "BATCH_VARIANCE", "UNINFORMATIVE",
         "DROPPED", "UNPROCESSED"]
FILED = {"BI_BATCH_DEPENDENT", "BI_SCHEDULE_DEPENDENT", "BI_RELATION_PERSISTS",
         "SOLO_NONDETERMINISTIC"}


def find_logs(paths: list[str]) -> list[Path]:
    logs: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            logs.append(p)
        elif p.is_dir():
            logs.extend(sorted(p.rglob("confirmation_log.jsonl")))
    return logs


def tally(log: Path) -> dict:
    counts: Counter = Counter()
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        v, prio = rec.get("verdict", "?"), rec.get("priority", "?")
        counts[v] += 1
        counts[f"{v}|{prio}"] += 1
        counts[f"all|{prio}"] += 1
        counts["all"] += 1
        if v in FILED:
            counts["filed"] += 1
            counts[f"filed|{prio}"] += 1
    summary = log.with_name("confirmation_summary.json")
    if summary.exists():
        s = json.loads(summary.read_text()).get("counts", {})
        counts["cooled"] = s.get("cooled", 0)
    return {"run": str(log.parent), "counts": dict(counts)}


def render(rows: list[dict]) -> str:
    verdicts = [v for v in ORDER if any(r["counts"].get(v) for r in rows)]
    head = ["run", "logged", "filed"] + verdicts + ["cooled"]
    out = [" | ".join(head), " | ".join("---" for _ in head)]
    for r in rows:
        c = r["counts"]
        cells = [r["run"], str(c.get("all", 0)), str(c.get("filed", 0))]
        for v in verdicts:
            cells.append(f"{c.get(v, 0)} ({c.get(v + '|nontie', 0)}/{c.get(v + '|tie', 0)})")
        cells.append(str(c.get("cooled", 0)))
        out.append(" | ".join(cells))
    out.append("")
    out.append("cells: total (nontie/tie); 'filed' = verdicts filed as findings;"
               " 'cooled' = candidates never enqueued (signature cooldown)")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    logs = find_logs(args.paths)
    if not logs:
        print("no confirmation_log.jsonl found", file=sys.stderr)
        return 1
    rows = [tally(l) for l in logs]
    total: Counter = Counter()
    for r in rows:
        total.update(r["counts"])
    if len(rows) > 1:
        rows.append({"run": "TOTAL", "counts": dict(total)})
    print(json.dumps(rows, indent=2) if args.json else render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
