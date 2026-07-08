import pytest
import config


def test_peak_advances_on_growth(tmp_path, monkeypatch):
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert risk_state.get_peak_value(100_000) == pytest.approx(100_000)
    assert risk_state.get_peak_value(110_000) == pytest.approx(110_000)  # new high
    assert risk_state.get_peak_value(105_000) == pytest.approx(110_000)  # drawdown — peak stays


def test_peak_persists_across_instances(tmp_path, monkeypatch):
    from src.execution import risk_state
    import importlib

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    risk_state.get_peak_value(120_000)

    # Reload module to simulate a fresh daily_run.py process
    importlib.reload(risk_state)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # re-apply after reload
    peak = risk_state.get_peak_value(115_000)  # below prior peak
    assert peak == pytest.approx(120_000)


# ---------------------------------------------------------------------------
# Halt flag (#14)
# ---------------------------------------------------------------------------

def test_set_halted_then_is_halted_true(tmp_path, monkeypatch):
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert risk_state.is_halted("2026-06-08") is False

    risk_state.set_halted("2026-06-08")

    assert risk_state.is_halted("2026-06-08") is True
    # A different date must not be considered halted.
    assert risk_state.is_halted("2026-06-09") is False


def test_halt_flag_persists_across_instances(tmp_path, monkeypatch):
    from src.execution import risk_state
    import importlib

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    risk_state.set_halted("2026-06-08")

    importlib.reload(risk_state)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    assert risk_state.is_halted("2026-06-08") is True


def test_set_halted_does_not_clobber_peak_value(tmp_path, monkeypatch):
    """set_halted() must not lose the peak_value that get_peak_value() tracks
    (they share the same risk_state.json file)."""
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    risk_state.get_peak_value(120_000)
    risk_state.set_halted("2026-06-08")

    assert risk_state.get_peak_value(50_000) == pytest.approx(120_000)
    assert risk_state.is_halted("2026-06-08") is True


# ---------------------------------------------------------------------------
# Corrupt-file safety (#17a)
# ---------------------------------------------------------------------------

def test_corrupt_risk_state_is_preserved_not_wiped(tmp_path, monkeypatch):
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    path = tmp_path / "risk_state.json"
    path.write_text("{this is not valid json!!")

    # Reading through get_peak_value must not crash, and must start fresh...
    peak = risk_state.get_peak_value(50_000)
    assert peak == pytest.approx(50_000)

    # ...but the corrupt original must be preserved under a renamed file,
    # never silently discarded.
    corrupt_files = list(tmp_path.glob("risk_state.json.corrupt-*"))
    assert len(corrupt_files) == 1, f"Expected exactly one preserved corrupt file, got {corrupt_files}"
    assert corrupt_files[0].read_text() == "{this is not valid json!!"


# ---------------------------------------------------------------------------
# Atomic write safety (#17a)
# ---------------------------------------------------------------------------

def test_atomic_write_failure_preserves_old_file(tmp_path, monkeypatch):
    """If the write fails partway through (simulated os.replace failure), the
    previously-written file must be left completely intact — never a
    half-written / corrupt file."""
    from src.execution import risk_state

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    risk_state.get_peak_value(100_000)  # establish a known-good file

    path = tmp_path / "risk_state.json"
    original_content = path.read_text()

    def boom(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(risk_state.os, "replace", boom)

    with pytest.raises(OSError):
        risk_state.get_peak_value(999_000)

    # Old file must be untouched (still readable, still the OLD value).
    assert path.read_text() == original_content
    # No leftover temp file should remain behind.
    assert list(tmp_path.glob("risk_state.json.*.tmp")) == []
