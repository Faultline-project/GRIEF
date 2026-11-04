"""
Tests for scripts/sglang_repro_flags.py

Covers:
  - extract_server_flags: stripping script-managed flags, deduplication,
    missing/empty server_flags, non-existent finding path
  - strip_lora: removing --lora-paths with various name=path patterns
  - strip_spec: removing speculative-algorithm/num-steps/draft-model-path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import module under test directly (scripts/ is not a package)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from sglang_repro_flags import extract_server_flags, strip_lora, strip_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_finding(tmp_path: Path, server_flags: str | None) -> str:
    d = {"bug_class": "kv_corruption", "score": 7.0}
    if server_flags is not None:
        d["server_flags"] = server_flags
    p = tmp_path / "finding.json"
    p.write_text(json.dumps(d))
    return str(p)


# ---------------------------------------------------------------------------
# extract_server_flags
# ---------------------------------------------------------------------------

class TestExtractServerFlags:
    def test_returns_empty_for_missing_server_flags(self, tmp_path):
        p = _write_finding(tmp_path, server_flags=None)
        assert extract_server_flags(p) == ""

    def test_returns_empty_for_none_string(self, tmp_path):
        p = _write_finding(tmp_path, "NONE")
        assert extract_server_flags(p) == ""

    def test_returns_empty_for_whitespace_only(self, tmp_path):
        p = _write_finding(tmp_path, "   ")
        assert extract_server_flags(p) == ""

    def test_returns_empty_for_missing_file(self):
        assert extract_server_flags("/nonexistent/path/finding.json") == ""

    def test_strips_model_flag(self, tmp_path):
        p = _write_finding(tmp_path, "--model Qwen/Qwen2.5-0.5B --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--model" not in result
        assert "Qwen/Qwen2.5-0.5B" not in result

    def test_strips_port_flag(self, tmp_path):
        p = _write_finding(tmp_path, "--port 8000 --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--port" not in result
        assert "8000" not in result

    def test_strips_mem_fraction_static(self, tmp_path):
        p = _write_finding(tmp_path, "--mem-fraction-static 0.88 --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--mem-fraction-static" not in result

    def test_strips_served_model_name(self, tmp_path):
        p = _write_finding(tmp_path, "--served-model-name mymodel --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--served-model-name" not in result

    def test_preserves_unmanaged_flags(self, tmp_path):
        p = _write_finding(tmp_path, "--enable-prefix-caching --max-running-requests 512")
        result = extract_server_flags(p)
        assert "--enable-prefix-caching" in result
        assert "--max-running-requests" in result
        assert "512" in result

    def test_preserves_lora_paths(self, tmp_path):
        p = _write_finding(tmp_path, "--lora-paths lora_a=/tmp/a lora_b=/tmp/b")
        result = extract_server_flags(p)
        assert "--lora-paths" in result
        assert "lora_a=/tmp/a" in result

    def test_preserves_speculative_flags(self, tmp_path):
        p = _write_finding(tmp_path, "--speculative-algorithm NEXTN --speculative-num-steps 5")
        result = extract_server_flags(p)
        assert "--speculative-algorithm" in result
        assert "NEXTN" in result
        assert "--speculative-num-steps" in result

    def test_deduplicates_repeated_flags(self, tmp_path):
        p = _write_finding(tmp_path,
            "--enable-prefix-caching --enable-prefix-caching --max-running-requests 512")
        result = extract_server_flags(p)
        assert result.count("--enable-prefix-caching") == 1

    def test_strips_multiple_managed_flags_together(self, tmp_path):
        p = _write_finding(tmp_path,
            "--model mymodel --port 9000 --mem-fraction-static 0.9 --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--model" not in result
        assert "--port" not in result
        assert "--mem-fraction-static" not in result
        assert "--enable-prefix-caching" in result

    def test_full_sglang_spec_finding(self, tmp_path):
        flags = (
            "--model Qwen/Qwen2.5-0.5B --port 8001 "
            "--speculative-algorithm NEXTN --speculative-num-steps 5 "
            "--enable-prefix-caching"
        )
        p = _write_finding(tmp_path, flags)
        result = extract_server_flags(p)
        assert "--model" not in result
        assert "--port" not in result
        assert "--speculative-algorithm NEXTN" in result
        assert "--speculative-num-steps 5" in result
        assert "--enable-prefix-caching" in result

    def test_full_sglang_lora_finding(self, tmp_path):
        flags = (
            "--model Qwen/Qwen2.5-0.5B --port 8001 "
            "--lora-paths lora_a=/scratch/lora_a lora_b=/scratch/lora_b"
        )
        p = _write_finding(tmp_path, flags)
        result = extract_server_flags(p)
        assert "--model" not in result
        assert "--lora-paths" in result
        assert "lora_a=/scratch/lora_a" in result
        assert "lora_b=/scratch/lora_b" in result

    # --flag=value form (inline value, no separate token)
    def test_strips_model_inline_value(self, tmp_path):
        p = _write_finding(tmp_path, "--model=Qwen/Qwen2.5-0.5B --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--model" not in result
        assert "Qwen/Qwen2.5-0.5B" not in result
        assert "--enable-prefix-caching" in result

    def test_strips_port_inline_value(self, tmp_path):
        p = _write_finding(tmp_path, "--port=9001 --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--port" not in result
        assert "9001" not in result

    def test_strips_mem_fraction_static_inline_value(self, tmp_path):
        p = _write_finding(tmp_path, "--mem-fraction-static=0.9 --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--mem-fraction-static" not in result
        assert "--enable-prefix-caching" in result

    def test_preserves_non_managed_inline_value(self, tmp_path):
        p = _write_finding(tmp_path, "--max-running-requests=256 --enable-prefix-caching")
        result = extract_server_flags(p)
        assert "--max-running-requests=256" in result
        assert "--enable-prefix-caching" in result

    def test_deduplicates_inline_value_form(self, tmp_path):
        p = _write_finding(tmp_path,
            "--enable-prefix-caching --max-running-requests=256 --max-running-requests=512")
        result = extract_server_flags(p)
        assert result.count("--max-running-requests") == 1


# ---------------------------------------------------------------------------
# strip_lora
# ---------------------------------------------------------------------------

class TestStripLora:
    def test_removes_lora_paths_with_single_adapter(self):
        result = strip_lora("--lora-paths lora_a=/tmp/a --enable-prefix-caching")
        assert "--lora-paths" not in result
        assert "lora_a=/tmp/a" not in result
        assert "--enable-prefix-caching" in result

    def test_removes_lora_paths_with_two_adapters(self):
        result = strip_lora("--lora-paths lora_a=/tmp/a lora_b=/tmp/b --enable-prefix-caching")
        assert "--lora-paths" not in result
        assert "lora_a=/tmp/a" not in result
        assert "lora_b=/tmp/b" not in result
        assert "--enable-prefix-caching" in result

    def test_noop_when_no_lora_paths(self):
        flags = "--enable-prefix-caching --max-running-requests 256"
        assert strip_lora(flags) == flags

    def test_noop_on_empty_string(self):
        assert strip_lora("") == ""

    def test_preserves_flags_after_lora_paths(self):
        result = strip_lora("--lora-paths lora_a=/tmp/a --speculative-algorithm NEXTN")
        assert "--speculative-algorithm" in result
        assert "NEXTN" in result

    def test_preserves_flags_before_lora_paths(self):
        result = strip_lora("--enable-prefix-caching --lora-paths lora_a=/tmp/a")
        assert "--enable-prefix-caching" in result

    def test_no_extra_whitespace(self):
        result = strip_lora("--enable-prefix-caching --lora-paths lora_a=/tmp/a --port 8000")
        # Should not produce double spaces (collapsed by join/split)
        assert "  " not in result

    def test_does_not_strip_non_lora_paths_flag(self):
        """--lora-something-else should not be stripped."""
        flags = "--lora-modules somemod --enable-prefix-caching"
        result = strip_lora(flags)
        assert "--lora-modules" in result

    def test_removes_lora_paths_inline_value(self):
        """--lora-paths=name=/path (unusual but valid inline form)."""
        result = strip_lora("--lora-paths=lora_a=/tmp/a --enable-prefix-caching")
        assert "--lora-paths" not in result
        assert "--enable-prefix-caching" in result


# ---------------------------------------------------------------------------
# strip_spec
# ---------------------------------------------------------------------------

class TestStripSpec:
    def test_removes_all_three_spec_flags(self):
        """Subsumes the four per-flag space-form tests: it also asserts that
        each flag's *value* is removed with it, which is what those tests
        added over this one."""
        flags = (
            "--speculative-algorithm NEXTN "
            "--speculative-num-steps 5 "
            "--speculative-draft-model-path /tmp/eagle "
            "--enable-prefix-caching"
        )
        result = strip_spec(flags)
        for gone in ("--speculative-algorithm", "--speculative-num-steps",
                     "--speculative-draft-model-path",
                     "NEXTN", "/tmp/eagle"):
            assert gone not in result, gone
        assert "--enable-prefix-caching" in result

    def test_noop_when_no_spec_flags(self):
        flags = "--enable-prefix-caching --max-running-requests 256"
        assert strip_spec(flags) == flags

    def test_noop_on_empty_string(self):
        assert strip_spec("") == ""

    def test_preserves_lora_flags(self):
        flags = "--lora-paths lora_a=/tmp/a --speculative-algorithm NEXTN"
        result = strip_spec(flags)
        assert "--lora-paths" in result
        assert "lora_a=/tmp/a" in result
        assert "--speculative-algorithm" not in result

    def test_no_extra_whitespace(self):
        result = strip_spec(
            "--enable-prefix-caching --speculative-algorithm NEXTN --speculative-num-steps 5"
        )
        assert "  " not in result

    def test_removes_all_three_inline_value_form(self):
        """Subsumes the three per-flag `--flag=value` tests, values included."""
        flags = (
            "--speculative-algorithm=EAGLE "
            "--speculative-num-steps=5 "
            "--speculative-draft-model-path=/tmp/eagle "
            "--enable-prefix-caching"
        )
        result = strip_spec(flags)
        for gone in ("--speculative-algorithm", "--speculative-num-steps",
                     "--speculative-draft-model-path",
                     "EAGLE", "/tmp/eagle"):
            assert gone not in result, gone
        assert "--enable-prefix-caching" in result


# ---------------------------------------------------------------------------
# CLI smoke tests (subprocess)
# ---------------------------------------------------------------------------

class TestCLI:
    """Verify the module runs correctly as a CLI script."""

    def _run(self, *args) -> str:
        import subprocess
        script = str(Path(__file__).parent.parent / "scripts" / "sglang_repro_flags.py")
        result = subprocess.run(
            [sys.executable, script, *args],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_extract_server_flags_missing_file_cli(self):
        import subprocess
        script = str(Path(__file__).parent.parent / "scripts" / "sglang_repro_flags.py")
        result = subprocess.run(
            [sys.executable, script, "extract-server-flags", "/no/such/file.json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_unknown_command_nonzero_exit(self):
        import subprocess
        script = str(Path(__file__).parent.parent / "scripts" / "sglang_repro_flags.py")
        result = subprocess.run(
            [sys.executable, script, "no-such-command"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
