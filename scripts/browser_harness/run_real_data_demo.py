"""使用真实1000商品快照演示完整浏览器购物闭环并生成截图。"""

from __future__ import annotations

import json
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, redirect, request
from markupsafe import escape
from werkzeug.serving import make_server

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from browser_harness import FaultPlan, HarnessConfig, PlaywrightWebShopHarness  # noqa: E402


DATA_DIR = (
    REPO_ROOT
    / "agent_system"
    / "environments"
    / "env_package"
    / "webshop"
    / "webshop"
    / "data"
)
TARGET_ASIN = "B005X7IXTK"
PAGE_SIZE = 8

STYLE = """
<style>
body{margin:0;background:#f4f6f8;color:#172033;font-family:Arial,Microsoft YaHei,sans-serif}
.top{background:#131921;color:white;padding:16px 6%;font-size:25px;font-weight:700}
.subtop{background:#232f3e;color:white;padding:10px 6%}.wrap{max-width:1180px;margin:28px auto;padding:0 22px}
.task{background:#fff8e6;border-left:5px solid #f59e0b;padding:17px 20px;margin-bottom:20px;line-height:1.6}
.search{display:flex;gap:0;margin-bottom:25px}.search input{flex:1;padding:14px;border:2px solid #f59e0b;font-size:17px}
.search button,.btn{background:#f59e0b;color:#111827;border:0;padding:13px 22px}
.search button,.btn{font-weight:700;text-decoration:none;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
.product{background:white;border:1px solid #d5d9d9;padding:15px;min-height:390px}
.product img{width:100%;height:180px;object-fit:contain;background:white}
.product h3{font-size:16px;line-height:1.4;height:68px;overflow:hidden}
.asin{display:inline-block;color:#2162a1;font-weight:700;text-decoration:none;margin-top:8px}
.price{font-size:22px;font-weight:700;margin:8px 0}.muted{color:#64748b}
.badge{display:inline-block;background:#e8f1fb;color:#24549a;padding:5px 9px;margin:3px;border-radius:12px}
input:checked + .badge{background:#24549a;color:white;box-shadow:0 0 0 2px #93c5fd}
.detail{background:white;border:1px solid #d5d9d9;padding:26px;display:grid}
.detail{grid-template-columns:38% 62%;gap:28px}.detail img{width:100%;height:360px;object-fit:contain}
</style>
"""


def _load_data():
    items = json.loads((DATA_DIR / "items_shuffle_1000.json").read_text(encoding="utf-8"))
    goals = json.loads((DATA_DIR / "items_human_ins.json").read_text(encoding="utf-8"))
    item_by_asin = {item["asin"]: item for item in items}
    return items, item_by_asin, goals[TARGET_ASIN][0]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _rank(items: list[dict], query: str) -> list[dict]:
    query_tokens = _tokens(query)

    def score(item: dict) -> tuple[int, str]:
        text = " ".join(
            str(item.get(field, ""))
            for field in ("name", "full_description", "product_category", "query", "small_description")
        ).casefold()
        lexical_score = sum(text.count(token) for token in query_tokens)
        return lexical_score, item["asin"]

    return sorted(items, key=score, reverse=True)


def create_real_data_app() -> Flask:
    items, item_by_asin, goal = _load_data()
    app = Flask(__name__)

    def shell(content: str) -> str:
        return (
            f"<html><head>{STYLE}</head><body>"
            "<div class='top'>WebShop · Amazon商品快照</div>"
            "<div class='subtop'>Browser Harness真实数据演示｜1000件商品</div>"
            f"<div class='wrap'>{content}</div></body></html>"
        )

    def task_box() -> str:
        instruction = escape(goal["instruction"])
        return f"<div id='instruction-text' class='task'><b>用户购物任务</b><br>{instruction}</div>"

    @app.route("/<session_id>", methods=["GET", "POST"])
    def home(session_id: str):
        if request.method == "POST":
            query = request.form["search_query"]
            return redirect(f"/search_results/{session_id}/1?q={query}")
        return shell(
            task_box()
            + "<form class='search' method='post'>"
            + "<input id='search_input' name='search_query' placeholder='Search products'>"
            + "<button type='submit'>Search</button></form>"
        )

    @app.route("/search_results/<session_id>/<int:page>")
    def results(session_id: str, page: int):
        query = request.args.get("q", "")
        ranked = _rank(items, query)
        start = (page - 1) * PAGE_SIZE
        cards = []
        for item in ranked[start : start + PAGE_SIZE]:
            image = (item.get("images") or [""])[0]
            cards.append(
                "<article class='product'>"
                f"<img src='{escape(image)}' alt='product image'>"
                f"<a class='product-link asin' href='/item_page/{session_id}/{item['asin']}?q={escape(query)}'>"
                f"{item['asin']}</a>"
                f"<h3>{escape(item.get('name', ''))}</h3>"
                f"<div class='price'>{escape(str(item.get('pricing') or 'Price unavailable'))}</div>"
                f"<div class='muted'>{escape(str(item.get('average_rating') or 'No rating'))} stars</div>"
                "</article>"
            )
        next_url = f"/search_results/{session_id}/{page + 1}?q={escape(query)}"
        content = (
            task_box()
            + f"<h2>搜索“{escape(query)}”</h2>"
            + f"<p class='muted'>第{page}页 · 当前展示{len(cards)}个候选 · 来自真实1000商品数据</p>"
            + "<div class='grid'>"
            + "".join(cards)
            + "</div>"
            + f"<p><a class='btn' href='{next_url}'>Next &gt;</a></p>"
        )
        return shell(content)

    @app.route("/item_page/<session_id>/<asin>")
    def item_page(session_id: str, asin: str):
        item = item_by_asin[asin]
        image = (item.get("images") or [""])[0]
        query = request.args.get("q", "")
        selected = {
            key: value
            for key, value in request.args.items()
            if key != "q"
        }
        options = []
        for option_name, choices in (item.get("customization_options") or {}).items():
            option_controls = []
            for option_index, choice in enumerate(choices[:8]):
                value = choice.get("value", "")
                params = {"q": query, **selected, option_name: value}
                option_url = f"/item_page/{session_id}/{asin}?{urlencode(params)}"
                checked = " checked" if selected.get(option_name) == value else ""
                control_id = f"option-{len(options)}-{option_index}"
                option_controls.append(
                    f"<input type='radio' id='{control_id}' name='{escape(option_name)}' "
                    f"value='{escape(value)}'{checked} style='display:none'>"
                    f"<label class='badge' for='{control_id}' "
                    f"onclick=\"window.location.href='{escape(option_url)}';return false;\">"
                    f"{escape(value)}</label>"
                )
            options.append(f"<h4>{escape(option_name)}</h4>" + "".join(option_controls))
        hidden_options = "".join(
            f"<input type='hidden' name='{escape(key)}' value='{escape(value)}'>"
            for key, value in selected.items()
        )
        content = (
            task_box()
            + "<section class='detail'>"
            + f"<img src='{escape(image)}' alt='product image'><div>"
            + f"<div class='asin'>{asin}</div><h2>{escape(item.get('name', ''))}</h2>"
            + f"<div class='price'>{escape(str(item.get('pricing') or 'Price unavailable'))}</div>"
            + f"<p>{escape(item.get('full_description') or '')}</p>"
            + "".join(options)
            + f"<form method='post' action='/done/{session_id}/{asin}'>"
            + hidden_options
            + "<button type='submit'>Buy Now</button></form></div></section>"
        )
        return shell(content)

    @app.route("/done/<session_id>/<asin>", methods=["POST"])
    def done(session_id: str, asin: str):
        required = {}
        for raw_option in goal.get("options", []):
            key, separator, value = raw_option.partition(":")
            if separator:
                required[key.strip().casefold()] = value.strip().casefold()
        selected = {
            key.strip().casefold(): value.strip().casefold()
            for key, value in request.form.items()
        }
        option_match = all(selected.get(key) == value for key, value in required.items())
        reward = 1.0 if asin == TARGET_ASIN and option_match else 0.0
        status = "购买任务完成" if reward == 1.0 else "购买结果与任务约束不匹配"
        return shell(
            f"<div class='detail'><div><div id='thankyou'><h1>{status}</h1></div>"
            f"<h2>{asin}</h2><p>所选规格：{escape(str(dict(request.form)))}</p>"
            f"<div id='reward'><b>任务奖励</b><pre>{reward}</pre></div></div>"
            "<div><h3>Browser Harness执行记录</h3>"
            "<p>购买经过显式确认，并由幂等保护防止重复提交。</p></div></div>"
        )

    return app


@contextmanager
def serve_demo():
    server = make_server("127.0.0.1", 0, create_real_data_app())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=3)


def main() -> None:
    trace = Path("results/browser_harness/real_data_demo.jsonl")
    screenshots = Path("results/browser_harness/screenshots")
    if trace.exists():
        trace.unlink()
    screenshots.mkdir(parents=True, exist_ok=True)
    for screenshot in screenshots.glob("*.png"):
        screenshot.unlink()

    with serve_demo() as base_url:
        config = HarnessConfig(
            base_url=base_url,
            browser_channel="msedge",
            headless=False,
            block_external_resources=False,
            demo_event_pause_seconds=1,
            screenshot_dir=screenshots,
            trace_path=trace,
        )
        faults = FaultPlan.from_spec("rate_limit=1;login=2;timeout=3")
        faults.retry_after_seconds = 0.5
        with PlaywrightWebShopHarness(config, faults) as harness:
            observation = harness.reset(task_index=0)
            print(json.dumps(observation.to_dict(), ensure_ascii=False, indent=2))
            result = harness.step("search[cruelty free deodorant woody scent]")
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            harness.step(f"click[{TARGET_ASIN}]")
            harness.step("click[woody scent]")
            harness.step("click[1.6 Ounce (Pack of 1)]")
            purchase = harness.step("click[buy now]", confirm_purchase=True)
            print(json.dumps(purchase.to_dict(), ensure_ascii=False, indent=2))
            input("\n关键场景截图已生成。检查浏览器或图片后，在此终端按回车关闭。")

    print(f"Trace已保存到：{trace.resolve()}")
    print(f"截图已保存到：{screenshots.resolve()}")


if __name__ == "__main__":
    main()
