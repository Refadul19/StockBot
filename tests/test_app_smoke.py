"""The dashboard must at least render without an exception. It runs offline against a stub fetch."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from test_jp_plan import synth  # noqa: E402


def _run(code="6857"):
    jp, sox = synth(n=600)
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    import jpgap.data as data

    orig = data.load_daily
    # serve synthetic data instead of the network
    def fake(ticker, start, end, cfg, refresh=False, offline=False):
        return sox.to_frame("close") if ticker == "^SOX" else jp
    data.load_daily = fake
    try:
        at.run()
        at.text_input[0].set_value(code)
        at.button[0].click().run() if at.button else at.run()
    finally:
        data.load_daily = orig
    return at


def test_dashboard_renders_levels_for_a_valid_code():
    at = _run("1111")
    assert not at.exception
    assert any("Reference target" in m.label for m in at.metric)
    assert any("not financial advice" in w.value for w in at.warning)


def test_dashboard_rejects_bad_input_politely():
    at = _run("AAPL")
    assert not at.exception
    assert any("Tokyo stock code" in e.value for e in at.error)
