"""启动不依赖商品数据的Browser Harness轻量可视化演示。"""

from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, redirect, request
from werkzeug.serving import make_server

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from browser_harness import FaultPlan, HarnessConfig, PlaywrightWebShopHarness  # noqa: E402


STYLE = """
<style>
body{margin:0;background:#f4f7fb;color:#172033;font-family:Segoe UI,Microsoft YaHei,sans-serif}
.top{background:#24549a;color:white;padding:18px 7%;font-size:24px;font-weight:700}
.wrap{max-width:1050px;margin:38px auto;padding:0 24px}
.card{background:white;border:1px solid #dbe4f0;border-radius:16px;padding:28px}
.card{box-shadow:0 12px 35px rgba(43,73,115,.08)}
.task{background:#edf5ff;border-left:5px solid #3478d4;padding:16px 18px;margin-bottom:24px;line-height:1.6}
input{width:70%;padding:13px;border:1px solid #9fb4cf;border-radius:8px;font-size:17px}
button,.btn{display:inline-block;background:#2d67b1;color:white;border:0;border-radius:8px}
button,.btn{padding:13px 22px;font-size:17px;text-decoration:none;cursor:pointer}
.product{display:flex;justify-content:space-between;align-items:center}
.product{border:1px solid #dbe4f0;border-radius:12px;padding:20px;margin:14px 0}
.price{font-size:22px;color:#b42318;font-weight:700}.muted{color:#64748b}
.ok{font-size:30px;color:#15803d;font-weight:700}
</style>
"""


def create_demo_app() -> Flask:
    app = Flask(__name__)

    def shell(content: str) -> str:
        return (
            f"<html><head>{STYLE}</head><body>"
            "<div class='top'>ShopPilot Browser Harness 演示站</div>"
            f"<div class='wrap'>{content}</div></body></html>"
        )

    @app.route("/<session_id>", methods=["GET", "POST"])
    def home(session_id: str):
        if request.method == "POST":
            return redirect(f"/search_results/{session_id}/{request.form['search_query']}/1")
        return shell("""
        <div class="card"><div id="instruction-text" class="task"><b>购物任务：</b>购买一双黑色、10码、价格低于80美元的跑鞋。</div>
        <h2>搜索商品</h2><form method="post">
        <input id="search_input" name="search_query" placeholder="输入搜索词">
        <button type="submit">Search</button></form></div>
        """)

    @app.route("/search_results/<session_id>/<query>/<int:page>")
    def results(session_id: str, query: str, page: int):
        return shell(f"""
        <div class="card"><div id="instruction-text" class="task"><b>购物任务：</b>购买一双黑色、10码、价格低于80美元的跑鞋。</div>
        <h2>“{query}”的搜索结果</h2><div class="muted">第 {page} 页 · 共2个候选</div>
        <div class="product"><div>
        <a class="product-link btn" href="/item_page/{session_id}/B001">B001</a>
        <h3>轻量防滑黑色跑鞋</h3><span class="muted">橡胶鞋底 · 日常跑步</span>
        </div><div class="price">$65.99</div></div>
        <div class="product"><div>
        <a class="product-link btn" href="/item_page/{session_id}/B002">B002</a>
        <h3>白色休闲步行鞋</h3><span class="muted">帆布鞋面 · 无10码</span>
        </div><div class="price">$72.00</div></div>
        <button type="button">Next &gt;</button></div>
        """)

    @app.route("/item_page/<session_id>/<asin>")
    def item(session_id: str, asin: str):
        return shell(f"""
        <div class="card"><div id="instruction-text" class="task"><b>购物任务：</b>购买一双黑色、10码、价格低于80美元的跑鞋。</div>
        <h2>{asin} · 轻量防滑黑色跑鞋</h2><div class="price">$65.99</div>
        <p>颜色：Black　尺码：10　材质：透气网布　鞋底：防滑橡胶</p>
        <form method="post" action="/done/{session_id}/{asin}"><button type="submit">Buy Now</button></form></div>
        """)

    @app.route("/done/<session_id>/<asin>", methods=["POST"])
    def done(session_id: str, asin: str):
        return shell(f"""
        <div class="card"><div id="thankyou" class="ok">购买任务完成</div><h2>{asin} · 黑色 · 10码</h2>
        <p>Browser Harness 已安全执行购买动作，未发生重复提交。</p><div id="reward"><b>任务奖励</b><pre>1.0</pre></div></div>
        """)

    return app


@contextmanager
def serve_demo():
    server = make_server("127.0.0.1", 0, create_demo_app())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=3)


def show_step(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    time.sleep(1.5)


def main() -> None:
    trace = Path("results/browser_harness/visual_demo.jsonl")
    if trace.exists():
        trace.unlink()

    with serve_demo() as base_url:
        config = HarnessConfig(
            base_url=base_url,
            browser_channel="msedge",
            headless=False,
            trace_path=trace,
            backoff_base_seconds=0.2,
            backoff_jitter_seconds=0,
            demo_event_pause_seconds=2.5,
        )
        faults = FaultPlan.from_spec("rate_limit=1;login=2;timeout=3")
        faults.retry_after_seconds = 0.5

        with PlaywrightWebShopHarness(config, faults) as harness:
            harness.reset(task_index=0)
            time.sleep(2)
            show_step("搜索：先展示429，再自动恢复", harness.step("search[black running shoes]"))
            show_step("点击商品：先展示登录过期，再自动恢复", harness.step("click[B001]"))
            show_step("购买：先展示超时，再进行安全重试", harness.step("click[buy now]", confirm_purchase=True))
            input("\n演示完成。查看浏览器和Trace后，回到此终端按回车关闭。")

    print(f"Trace已保存到：{trace.resolve()}")


if __name__ == "__main__":
    main()
