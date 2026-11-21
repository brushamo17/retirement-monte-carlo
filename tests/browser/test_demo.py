"""End-to-end tests of the browser demo (docs/index.html) in a real browser.

These are skipped by a plain ``pytest`` run, because they need Playwright, a
browser and (by default) internet access. Run them with::

    pip install playwright && playwright install chromium
    BROWSER_TESTS=1 pytest tests/browser -q                          # macOS / Linux
    $env:BROWSER_TESTS=1; pytest tests/browser -q                    # Windows PowerShell

Two backends:

* ``BROWSER_BACKEND=pyodide`` (default): the page loads Pyodide and NumPy from
  the CDN, exactly as visitors get it, and the tests check that the browser's
  numbers match running the same package in ordinary Python. This is the
  "browser and Python agree" reproducibility check; CI runs it on every push.
* ``BROWSER_BACKEND=cpython``: the Pyodide worker is swapped for a stand-in that
  sends each call to a local Python server. It tests the page itself (inputs,
  charts, buttons) without downloading Pyodide, which is handy offline.

Set ``SCREENSHOT_DIR`` to save screenshots of the page.
"""

from __future__ import annotations

import json
import math
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

if not os.environ.get("BROWSER_TESTS"):
    pytest.skip("browser tests are opt-in: set BROWSER_TESTS=1", allow_module_level=True)
playwright_api = pytest.importorskip("playwright.sync_api")

from retirement_mc import web  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"
BACKEND = os.environ.get("BROWSER_BACKEND", "pyodide")
SCREENSHOTS = os.environ.get("SCREENSHOT_DIR")
LOAD_TIMEOUT = 180_000 if BACKEND == "pyodide" else 30_000

# Stand-in for docs/pyodide-worker.js in cpython mode: same messages, but each call
# goes to the test server, which runs retirement_mc.web in ordinary Python.
STUB_WORKER = """
postMessage({ status: "test backend" });
postMessage({ ready: true });
onmessage = async ({ data }) => {
  const { id, fn, args } = data;
  try {
    const r = await fetch("/api/" + fn, { method: "POST", body: JSON.stringify(args || {}) });
    const text = await r.text();
    if (!r.ok) throw new Error(text);
    postMessage({ id, result: JSON.parse(text) });
  } catch (err) { postMessage({ id, error: String(err.message || err) }); }
};
"""


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        fn = self.path.rsplit("/", 1)[-1]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        try:
            out, code = web.call(fn, body).encode(), 200
        except Exception as err:  # report to the page like the real worker would
            out, code = repr(err).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(DOCS)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        exe = os.environ.get("CHROME_PATH")
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser, server):
    pg = browser.new_page(viewport={"width": 1360, "height": 1000})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "fonts.g" not in m.text else None)
    if BACKEND == "cpython":
        pg.route("**/pyodide-worker.js", lambda route: route.fulfill(
            status=200, content_type="text/javascript", body=STUB_WORKER))
    pg.goto(server + "/index.html")
    pg.wait_for_function("window.demo && document.getElementById('loading').hidden", timeout=LOAD_TIMEOUT)
    wait_idle(pg)
    yield pg
    assert not errors, f"browser errors: {errors}"
    pg.close()


def wait_idle(pg, timeout=60_000):
    """Wait until the page has finished every queued simulation (run, sensitivity, grid)."""
    pg.wait_for_timeout(300)
    pg.wait_for_function("demo.idle() && demo.results().grid !== null", timeout=timeout)


def shot(pg, name):
    if SCREENSHOTS:
        Path(SCREENSHOTS).mkdir(parents=True, exist_ok=True)
        pg.screenshot(path=str(Path(SCREENSHOTS) / f"{name}.png"), full_page=True)


def assert_close(a, b, path="result", rel=1e-9, abs_=1e-9):
    """Recursively compare two JSON results with a floating-point tolerance."""
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ"
        for k in a:
            assert_close(a[k], b[k], f"{path}.{k}", rel, abs_)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{i}]", rel, abs_)
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and math.isclose(a, b, rel_tol=rel, abs_tol=abs_), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def text(pg, el_id):
    return pg.locator("#" + el_id).inner_text()


# ---- the page reproduces Python ----------------------------------------------

def test_default_page_shows_the_readme_result(page):
    assert text(page, "tSucc") == "89.4%"
    assert "95% CI 88.8%–90.0%" in text(page, "tSuccSub")
    assert page.locator("#heat button").count() == 99
    shot(page, "01-default")


CASES = {
    "default": {},
    "annuity": next(s["changes"] for s in web.SCENARIOS if s["id"] == "annuity"),
    "pension_fat_tails": {**next(s["changes"] for s in web.SCENARIOS if s["id"] == "pension"),
                          "model": "fat_tailed", "n_paths": 5_000},
    "bootstrap_guardrails": {"model": "bootstrap", "rule": "guardrails", "n_paths": 5_000, "seed": 11},
    "antithetic_custom_market": {"antithetic": True, "market": "custom", "stock_mean": 0.06, "n_paths": 4_000},
}


@pytest.mark.parametrize("case", list(CASES))
def test_browser_numbers_match_python(page, case):
    """Same inputs and seed: the browser and ordinary Python give the same numbers.

    With the cpython backend this is trivially true; with Pyodide it checks the
    whole chain: NumPy's random streams, floating point in WebAssembly, the zip
    bundle, and JSON in and out. Tiny differences in the last digits of exp/log
    between platforms are allowed for; a different success count is not.
    """
    inputs = CASES[case]
    for fn, args in [("run", {"inputs": inputs}), ("sensitivity", {"inputs": inputs, "max_paths": 2_000}),
                     ("compare", {"a": {}, "b": inputs})]:
        browser_out = page.evaluate("([fn, args]) => demo.py(fn, args)", [fn, args])
        python_out = json.loads(web.call(fn, json.dumps(args)))
        assert_close(browser_out, python_out, f"{case}.{fn}", rel=1e-7, abs_=1e-7)


# ---- the page works -------------------------------------------------------------

def test_scenarios_and_income(page):
    page.select_option("#scenario", "annuity")
    wait_idle(page)
    expected = web.run(next(s["inputs"] for s in web.presets()["scenarios"] if s["id"] == "annuity"))
    assert text(page, "tSucc") == f"{expected['success']['p'] * 100:.1f}%"
    assert "Inflation-adjusted annuity" in text(page, "incomeLegend")
    assert page.locator("#income rect").count() > 30
    assert page.locator("#ssOn").is_checked() and not page.locator("#annPrem").is_hidden()
    assert "Pays $17,229 a year for life" in text(page, "annHint")
    shot(page, "02-annuity")

    page.locator("#ssAge").fill("70")
    page.locator("#ssAge").dispatch_event("change")
    wait_idle(page)
    assert page.locator("#scenario").input_value() == "custom"
    assert "$39,680" in text(page, "ssHint")


def test_compare_two_plans(page):
    page.select_option("#scenario", "ss67")
    wait_idle(page)
    page.click("#pinBtn")
    wait_idle(page)
    assert "identical" in text(page, "cmpInsight")
    page.select_option("#scenario", "ss70")
    wait_idle(page)
    page.wait_for_function("demo.results().compare && demo.results().compare.diff.p !== 0")
    expected = web.compare(next(s["inputs"] for s in web.presets()["scenarios"] if s["id"] == "ss67"),
                           next(s["inputs"] for s in web.presets()["scenarios"] if s["id"] == "ss70"))
    assert f"+{expected['diff']['p'] * 100:.1f} pts" in text(page, "cmpTable")
    assert "excludes zero" in text(page, "cmpInsight")
    shot(page, "03-compare")
    page.click("#clearBtn")
    assert page.locator("#cmpBody").is_hidden()


def test_sensitivity_and_heatmap(page):
    page.select_option("#scenario", "pension")
    wait_idle(page)
    names = [f["name"] for f in page.evaluate("demo.results().sensitivity.factors")]
    assert "Inflation" in names and "Longevity" in names
    assert page.locator("#tornado rect.trow").count() == len(names)
    shot(page, "04-pension")

    # With guaranteed income the columns center on the plan's own rate (15% here).
    assert "centered on your plan" in text(page, "heatCard")
    page.locator("#heat button[data-r='0.12'][data-s='0.5']").click()
    wait_idle(page)
    assert page.locator("#spending").input_value() == "48000"   # 12% of $400k
    assert page.locator("#stock").input_value() == "50"
    assert page.locator("#heat button.cur").count() == 1


def test_bad_input_does_not_break_the_page(page):
    page.locator("#spending").fill("")
    wait_idle(page)
    assert text(page, "tSucc").endswith("%")
    assert page.locator("#error").is_hidden()


def test_mobile_layout(browser, server):
    pg = browser.new_page(viewport={"width": 390, "height": 900})
    if BACKEND == "cpython":
        pg.route("**/pyodide-worker.js", lambda route: route.fulfill(
            status=200, content_type="text/javascript", body=STUB_WORKER))
    pg.goto(server + "/index.html")
    pg.wait_for_function("window.demo && document.getElementById('loading').hidden", timeout=LOAD_TIMEOUT)
    wait_idle(pg)
    overflow = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1
    shot(pg, "05-mobile")
    pg.close()
