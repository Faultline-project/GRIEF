#!/usr/bin/env python3
"""
scripts/sglang_repro_flags.py — SGLang server flag manipulation helpers.

Used when reproducing an SGLang finding to:
  - Extract server flags from a finding JSON (stripping script-managed flags)
  - Strip LoRA flags (--lora-paths) for base-model comparison runs
  - Strip speculative decoding flags for no-spec comparison runs

CLI:
    python3 scripts/sglang_repro_flags.py extract-server-flags <finding_path>
    python3 scripts/sglang_repro_flags.py strip-lora <flags_string>
    python3 scripts/sglang_repro_flags.py strip-spec <flags_string>
    python3 scripts/sglang_repro_flags.py strip-tp <flags_string>
"""

import json
import re
import shlex
import sys
from pathlib import Path

# Flags that the reproduction job sets explicitly — strip to avoid conflicts.
_SLURM_MANAGED = {
    "--model",
    "--model-path",         # canonical SGLang flag — slurm passes this explicitly
    "--port",
    "--mem-fraction-static",
    "--served-model-name",
    "--kv-events-config",   # reproduce script always adds its own with a per-task port
}


def extract_server_flags(finding_path: str) -> str:
    """Load finding JSON and return a cleaned server_flags string.

    Strips flags in _SLURM_MANAGED to avoid conflicts with the SLURM script's
    explicit --model, --port, --mem-fraction-static args.
    Returns empty string for findings with no server_flags recorded.
    Duplicate flags (same --flag appearing twice) are deduplicated: first wins.
    """
    try:
        d = json.loads(Path(finding_path).read_text())
    except Exception:
        d = {}

    raw = d.get("server_flags", "") or ""
    if raw.strip().upper() == "NONE":
        raw = ""
    if not raw.strip():
        return ""

    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()

    result = []
    seen_flags = set()
    skip_next = False
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        flag_base = tok.split("=")[0] if tok.startswith("--") else tok
        has_inline_value = tok != flag_base  # True for --flag=value form
        if flag_base in _SLURM_MANAGED:
            # For --flag=value the value is inline; no separate value token to skip.
            # For --flag value (space-separated), skip the next token if it's a value.
            if not has_inline_value and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                skip_next = True
            continue
        if tok.startswith("--"):
            if flag_base in seen_flags:
                if not has_inline_value and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                    skip_next = True
                continue
            seen_flags.add(flag_base)
        result.append(tok)

    return " ".join(result)


def strip_lora(flags: str) -> str:
    """Remove SGLang LoRA flags from *flags*.

    SGLang LoRA flag format::

        --lora-paths name1=/path/1 name2=/path/2

    Removes ``--lora-paths`` together with all following ``name=path`` tokens
    (i.e. tokens containing ``=`` that do not start with ``--``).
    """
    # Space-separated: --lora-paths name1=/path1 name2=/path2
    # Inline-value:    --lora-paths=name1=/path1 (unusual but valid)
    flags = re.sub(r"--lora-paths(?:=\S+|(?:\s+[^-\s]\S*)*)", "", flags)
    return " ".join(flags.split())


def strip_spec(flags: str) -> str:
    """Remove SGLang speculative-decoding flags from *flags*.

    Strips each of::

        --speculative-algorithm <ALG>
        --speculative-num-steps <N>
        --speculative-draft-model-path <PATH>

    together with their values.
    """
    flags = re.sub(r"--speculative-algorithm(?:=|\s+)\S+", "", flags)
    flags = re.sub(r"--speculative-num-steps(?:=|\s+)\S+", "", flags)
    flags = re.sub(r"--speculative-draft-model-path(?:=|\s+)\S+", "", flags)
    flags = re.sub(r"--speculative-eagle-topk(?:=|\s+)\S+", "", flags)
    flags = re.sub(r"--speculative-num-draft-tokens(?:=|\s+)\S+", "", flags)
    return " ".join(flags.split())


def strip_tp(flags: str) -> str:
    """Remove tensor-parallel flag (--tp / --tp-size / --tensor-parallel-size).

    Drops the flag entirely so the server defaults to TP=1 (single rank).
    Used for the TP=2-vs-TP=1 differential repro to prove a bug is multi-rank.
    """
    flags = re.sub(r"--tp(?:=|\s+)\S+", "", flags)
    flags = re.sub(r"--tp-size(?:=|\s+)\S+", "", flags)
    flags = re.sub(r"--tensor-parallel-size(?:=|\s+)\S+", "", flags)
    return " ".join(flags.split())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "extract-server-flags":
        if len(sys.argv) < 3:
            print(
                "Usage: sglang_repro_flags.py extract-server-flags <finding_path>",
                file=sys.stderr,
            )
            sys.exit(1)
        print(extract_server_flags(sys.argv[2]))
    elif cmd == "strip-lora":
        flags = sys.argv[2] if len(sys.argv) > 2 else ""
        print(strip_lora(flags))
    elif cmd == "strip-spec":
        flags = sys.argv[2] if len(sys.argv) > 2 else ""
        print(strip_spec(flags))
    elif cmd == "strip-tp":
        flags = sys.argv[2] if len(sys.argv) > 2 else ""
        print(strip_tp(flags))
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
