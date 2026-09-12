from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_and_runs_example() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Single report", "Batch CSV", "Experiment results"]

    app.button[0].click().run()
    app.button[1].click().run()

    assert not app.exception
    assert app.text_area[0].value == "APT39 has used PowerShell to execute malicious code."
    assert len(app.get("download_button")) == 2


def test_routing_threshold_updates_existing_prediction() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()

    app.selectbox[0].set_value(3).run()
    app.button[0].click().run()
    app.button[1].click().run()

    assert any("Auto-route" in message.value for message in app.success)

    app.slider[0].set_value(0.90).run()

    assert any("Analyst review recommended" in message.value for message in app.warning)
