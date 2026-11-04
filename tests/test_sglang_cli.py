"""Tests for fuzzer/backends/sglang/cli.py — flag registration and adapter dispatch."""
import argparse
import pytest

from fuzzer.backends.sglang.cli import register_args, build_adapter, startup_lines
from fuzzer.adapters.sglang.legacy import SglangBaseAdapter
from fuzzer.adapters.sglang_lora.adapter import SglangLoRAAdapter
from fuzzer.adapters.sglang_moe.adapter import SglangMoEAdapter
from fuzzer.adapters.sglang_spec.adapter import SglangSpecAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(*argv):
    """Build a parser with SGLang flags registered and parse *argv*."""
    p = argparse.ArgumentParser()
    register_args(p)
    return p.parse_args(list(argv))


# ---------------------------------------------------------------------------
# No --sglang → returns None
# ---------------------------------------------------------------------------

class TestNoSglang:
    def test_returns_none_without_sglang(self):
        args = _parse()
        assert build_adapter(args) is None


# ---------------------------------------------------------------------------
# Base mode
# ---------------------------------------------------------------------------

class TestBaseMode:
    def test_sglang_alone_returns_target_adapter(self):
        """After Phase 3, ``build_adapter()`` for the base SGLang surface
        returns the new-shape ``Adapter`` (Capabilities + Strategy
        umbrella) rather than the legacy class. The contract for the
        runner is that the returned object implements ``TargetAdapter``;
        anything beyond that is implementation detail."""
        from fuzzer.core.adapter import TargetAdapter
        args = _parse("--sglang")
        adapter = build_adapter(args)
        assert isinstance(adapter, TargetAdapter)

    def test_sglang_alone_exposes_capabilities_and_strategy(self):
        """Phase 3 marker: the base build is now a new-shape adapter, so
        it carries ``capabilities`` and ``strategy``. Legacy spec/lora/moe
        adapters do not (yet) — see Phase 4+."""
        args = _parse("--sglang")
        adapter = build_adapter(args)
        assert adapter.capabilities is not None
        assert adapter.strategy is not None

    def test_base_adapter_not_lora_moe_spec(self):
        args = _parse("--sglang")
        a = build_adapter(args)
        assert not isinstance(a, (SglangLoRAAdapter, SglangMoEAdapter, SglangSpecAdapter))


# ---------------------------------------------------------------------------
# LoRA mode
# ---------------------------------------------------------------------------

@pytest.fixture
def lora_dir(tmp_path):
    """A weights dir holding two adapter subdirs.

    ``build_adapter`` discovers adapters by scanning the directory on disk
    (cli.py:107-126), so these must really exist.
    """
    for name in ("lora_a", "lora_b"):
        (tmp_path / name).mkdir()
    return tmp_path


class TestLoRAMode:
    def test_sglang_lora_returns_lora_adapter(self, lora_dir):
        args = _parse("--sglang", "--sglang-lora",
                      "--sglang-lora-weights-dir", str(lora_dir))
        assert isinstance(build_adapter(args), SglangLoRAAdapter)

    def test_lora_adapter_has_correct_paths(self, lora_dir):
        args = _parse("--sglang", "--sglang-lora",
                      "--sglang-lora-weights-dir", str(lora_dir))
        a = build_adapter(args)
        assert a.adapters["lora_a"] == str(lora_dir / "lora_a")
        assert a.adapters["lora_b"] == str(lora_dir / "lora_b")

    def test_lora_discovers_only_present_subdirs(self, tmp_path):
        """Discovery is by scan, so lora_c..lora_f absent means absent."""
        for name in ("lora_a", "lora_b", "lora_c"):
            (tmp_path / name).mkdir()
        args = _parse("--sglang", "--sglang-lora",
                      "--sglang-lora-weights-dir", str(tmp_path))
        a = build_adapter(args)
        assert sorted(a.adapters) == ["lora_a", "lora_b", "lora_c"]

    def test_lora_takes_priority_over_spec(self, lora_dir):
        args = _parse("--sglang", "--sglang-lora", "--sglang-spec",
                      "--sglang-lora-weights-dir", str(lora_dir))
        assert isinstance(build_adapter(args), SglangLoRAAdapter)

    @pytest.mark.parametrize("names", [(), ("lora_a",)])
    def test_lora_raises_under_two_adapter_dirs(self, tmp_path, names):
        """The LoRA oracle compares outputs across adapters, so fewer than two
        is a silently useless campaign.  build_adapter must refuse to start."""
        for name in names:
            (tmp_path / name).mkdir()
        args = _parse("--sglang", "--sglang-lora",
                      "--sglang-lora-weights-dir", str(tmp_path))
        with pytest.raises(RuntimeError, match="at least 2 LoRA adapter dirs"):
            build_adapter(args)

    def test_lora_raises_when_weights_dir_missing(self, tmp_path):
        args = _parse("--sglang", "--sglang-lora",
                      "--sglang-lora-weights-dir", str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="at least 2 LoRA adapter dirs"):
            build_adapter(args)


# ---------------------------------------------------------------------------
# Spec mode
# ---------------------------------------------------------------------------

class TestSpecMode:
    def test_sglang_spec_returns_spec_adapter(self):
        args = _parse("--sglang", "--sglang-spec")
        assert isinstance(build_adapter(args), SglangSpecAdapter)

    def test_spec_default_algorithm_ngram(self):
        """NGRAM is the no-extra-weights default; NEXTN is not even a valid
        choice any more (it maps to EAGLE and needs a draft checkpoint)."""
        args = _parse("--sglang", "--sglang-spec")
        a = build_adapter(args)
        assert a.algorithm == "NGRAM"

    def test_spec_nextn_is_not_an_accepted_choice(self):
        with pytest.raises(SystemExit):
            _parse("--sglang", "--sglang-spec", "--sglang-spec-algorithm", "NEXTN")

    def test_spec_custom_num_steps(self):
        args = _parse("--sglang", "--sglang-spec", "--sglang-spec-num-steps", "8")
        a = build_adapter(args)
        assert a.num_steps == 8

    def test_spec_confirm_url_wired(self):
        args = _parse("--sglang", "--sglang-spec",
                      "--sglang-spec-confirm-url", "http://nospec:8001")
        a = build_adapter(args)
        assert a.confirm_url == "http://nospec:8001"

    def test_spec_eagle_with_draft_path(self):
        args = _parse("--sglang", "--sglang-spec",
                      "--sglang-spec-algorithm", "EAGLE",
                      "--sglang-spec-draft-model-path", "/tmp/eagle")
        a = build_adapter(args)
        assert a.algorithm == "EAGLE"
        assert a.draft_model_path == "/tmp/eagle"

    def test_spec_eagle_without_draft_path_raises(self):
        args = _parse("--sglang", "--sglang-spec",
                      "--sglang-spec-algorithm", "EAGLE")
        with pytest.raises(ValueError, match="draft_model_path"):
            build_adapter(args)

    def test_spec_takes_priority_over_moe(self):
        args = _parse("--sglang", "--sglang-spec", "--sglang-moe")
        assert isinstance(build_adapter(args), SglangSpecAdapter)


# ---------------------------------------------------------------------------
# MoE mode
# ---------------------------------------------------------------------------

class TestMoEMode:
    def test_sglang_moe_returns_moe_adapter(self):
        args = _parse("--sglang", "--sglang-moe")
        assert isinstance(build_adapter(args), SglangMoEAdapter)


# ---------------------------------------------------------------------------
# startup_lines
# ---------------------------------------------------------------------------

class TestStartupLines:
    """Only the confirm-url branch is covered here.

    The other five tests in this class pinned substrings of a human-readable
    log banner (cli.py:166-205) — they broke on wording changes and could not
    fail on a real regression, so they were removed.  These two stay because
    they assert an actual if/else in ``startup_lines`` whose silent wrong side
    is dangerous: a spec campaign with confirmation off produces findings that
    nothing ever confirms, and the banner is the only place that says so.
    """

    def test_spec_confirm_disabled_line_when_no_url(self):
        args = _parse("--sglang", "--sglang-spec")
        a = build_adapter(args)
        lines = startup_lines(args, a)
        assert any("disabled" in l for l in lines)

    def test_spec_confirm_enabled_line_when_url_set(self):
        args = _parse("--sglang", "--sglang-spec",
                      "--sglang-spec-confirm-url", "http://nospec:8001")
        a = build_adapter(args)
        lines = startup_lines(args, a)
        assert any("http://nospec:8001" in l for l in lines)
