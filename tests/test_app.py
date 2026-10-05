"""Smoke-test every page for every model and data source with Streamlit's AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["views/home.py", "views/model.py", "views/filter_step.py", "views/smoother.py",
         "views/likelihood.py", "views/diagnostics.py", "views/forecast.py",
         "views/discount.py", "views/bayes.py"]
MODELS = ["Local level", "Local linear trend", "Level + seasonal", "Dynamic regression"]


def _app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("source", ["Simulate", "Nile river (classic)"])
@pytest.mark.parametrize("model", MODELS)
def test_all_pages(source, model):
    at = _app()
    at.sidebar.radio(key="source").set_value(source).run()
    at.sidebar.selectbox(key="model_key").set_value(model).run()
    assert not at.exception, at.exception
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, f"{page}: {at.exception}"


def test_buttons_mle_gibbs_and_gap():
    at = _app()
    at.switch_page("views/likelihood.py").run()
    btn = [b for b in at.button if "maximum likelihood" in b.label][0]
    btn.click().run()
    assert not at.exception, at.exception
    use = [b for b in at.button if "Use these estimates" in b.label][0]
    use.click().run()
    assert not at.exception, at.exception
    v_after = at.sidebar.number_input(key="V_inf").value
    assert v_after != 1.0  # replaced by the MLE

    at.switch_page("views/bayes.py").run()
    at.select_slider(key="gn").set_value(200).run()
    [b for b in at.button if "Gibbs" in b.label][0].click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1

    at.switch_page("views/filter_step.py").run()
    at.slider(key="gap").set_value((40, 60)).run()
    at.checkbox(key="use_gap").check().run()
    at.slider(key="t_step").set_value(50).run()
    assert not at.exception, at.exception

    at.switch_page("views/diagnostics.py").run()
    at.select_slider(key="wfac").set_value(0.01).run()
    assert not at.exception, at.exception
