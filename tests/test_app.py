from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_and_runs_example() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Triage workspace", "Batch queue", "Model evidence"]

    app.button[1].click().run()
    app.button[0].click().run()

    assert not app.exception
    assert app.text_area[0].value == "APT39 has used PowerShell to execute malicious code."
    assert len(app.get("download_button")) == 1
