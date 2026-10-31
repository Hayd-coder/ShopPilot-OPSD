from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from flask import Flask, redirect, request
from werkzeug.serving import make_server

from browser_harness import FaultPlan, HarnessConfig, PlaywrightWebShopHarness
from browser_harness.models import ExecutionStatus


EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def _mini_webshop() -> Flask:
    app = Flask(__name__)

    @app.route("/<session_id>", methods=["GET", "POST"])
    def home(session_id: str):
        if request.method == "POST":
            return redirect(f"/search_results/{session_id}/{request.form['search_query']}/1")
        return """
        <html><body>
          <div id="instruction-text">Instruction: Find black running shoes under $80.</div>
          <form method="post"><input id="search_input" name="search_query">
          <button type="submit">Search</button></form>
        </body></html>
        """

    @app.route("/search_results/<session_id>/<query>/<int:page>")
    def results(session_id: str, query: str, page: int):
        return f"""
        <html><body>
          <div id="instruction-text">Instruction: Find black running shoes under $80.</div>
          <h1>Results for {query}</h1>
          <a class="product-link" href="/item_page/{session_id}/B001">B001</a>
          <button type="button">Next &gt;</button>
        </body></html>
        """

    @app.route("/item_page/<session_id>/<asin>")
    def item(session_id: str, asin: str):
        return f"""
        <html><body>
          <div id="instruction-text">Instruction: Find black running shoes under $80.</div>
          <h1>{asin} black running shoes</h1>
          <form method="post" action="/done/{session_id}/{asin}">
            <button type="submit">Buy Now</button>
          </form>
        </body></html>
        """

    @app.route("/done/<session_id>/<asin>", methods=["POST"])
    def done(session_id: str, asin: str):
        return """
        <html><body>
          <div id="thankyou">Thank you for shopping with us!</div>
          <div id="reward"><pre>1.0</pre></div>
        </body></html>
        """

    return app


@contextmanager
def _serve(app: Flask):
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=3)


@pytest.mark.skipif(os.name != "nt" or not EDGE.exists(), reason="local Edge channel is unavailable")
def test_real_browser_fault_recovery_and_purchase_guard(tmp_path: Path) -> None:
    with _serve(_mini_webshop()) as base_url:
        config = HarnessConfig(
            base_url=base_url,
            browser_channel="msedge",
            trace_path=tmp_path / "trace.jsonl",
            backoff_base_seconds=0,
            backoff_jitter_seconds=0,
        )
        faults = FaultPlan.from_spec("rate_limit=1;login=2")
        faults.retry_after_seconds = 0

        with PlaywrightWebShopHarness(config, faults) as harness:
            initial = harness.reset(task_index=0)
            assert initial.page_type == "search"

            search = harness.step("search[black running shoes]")
            assert search.status == ExecutionStatus.OK
            assert search.retries == 1
            assert search.observation.page_type == "search_results"

            product = harness.step("click[B001]")
            assert product.status == ExecutionStatus.OK
            assert product.observation.page_type == "item"

            blocked = harness.step("click[buy now]")
            assert blocked.status == ExecutionStatus.CONFIRMATION_REQUIRED

            purchase = harness.step("click[buy now]", confirm_purchase=True)
            assert purchase.status == ExecutionStatus.OK
            assert purchase.observation.done
            assert purchase.observation.reward == 1.0

        trace = (tmp_path / "trace.jsonl").read_text(encoding="utf-8")
        assert '"event": "rate_limited"' in trace
        assert '"event": "login_recovered"' in trace
        assert "not-a-real-secret" not in trace
