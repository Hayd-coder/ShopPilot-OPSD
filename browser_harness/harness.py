from __future__ import annotations

import hashlib
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .actions import ParsedAction, match_legal_action, parse_action
from .faults import FaultKind, FaultPlan
from .models import ActionResult, BrowserObservation, ExecutionStatus, HarnessConfig
from .trace import JsonlTraceWriter


class PlaywrightWebShopHarness:
    """在真实浏览器中执行现有的ShopPilot动作协议。

    网页基础设施异常与Agent任务失败分别记录。故障在浏览器动作执行前注入，
    从而保证恢复重试不会重复执行已经生效的购买动作。
    """

    def __init__(self, config: HarnessConfig | None = None, faults: FaultPlan | None = None):
        self.config = config or HarnessConfig()
        self.faults = faults or FaultPlan()
        self.trace = JsonlTraceWriter(Path(self.config.trace_path))
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._session_id = ""
        self._logical_step = 0
        self._side_effects: dict[str, str] = {}

    def __enter__(self) -> "PlaywrightWebShopHarness":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def start(self) -> "PlaywrightWebShopHarness":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install -r requirements-browser.txt"
            ) from exc

        self._playwright = sync_playwright().start()
        launch_options: dict[str, Any] = {"headless": self.config.headless}
        if self.config.browser_channel:
            launch_options["channel"] = self.config.browser_channel
        self._browser = self._playwright.chromium.launch(**launch_options)
        self._context = self._browser.new_context(service_workers="block")
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.config.action_timeout_ms)
        if self.config.block_external_resources:
            self._page.route("**/*", self._route_request)
        self.trace.write("harness_started", base_url=self.config.base_url)
        return self

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._page = self._context = self._browser = self._playwright = None

    def reset(self, task_index: int = 0, session_id: str | None = None) -> BrowserObservation:
        self._ensure_started()
        self._session_id = session_id or f"fixed_{task_index}"
        self._logical_step = 0
        self._side_effects.clear()
        url = f"{self.config.base_url.rstrip('/')}/{self._session_id}"
        response = self._page.goto(url, wait_until="domcontentloaded")
        if response is not None and response.status >= 400:
            raise RuntimeError(f"WebShop reset failed with HTTP {response.status}: {url}")
        observation = self.observe()
        self._capture_screenshot("00_task_search_page")
        self.trace.write("reset", observation=observation.to_dict())
        return observation

    def observe(self) -> BrowserObservation:
        self._ensure_started()
        body_text = self._clean_text(self._page.locator("body").inner_text())
        task_locator = self._page.locator("#instruction-text")
        task = self._clean_text(task_locator.inner_text()) if task_locator.count() else ""
        task = re.sub(r"^Instruction:\s*", "", task, flags=re.IGNORECASE)

        actions: list[str] = []
        search_input = self._page.locator("#search_input")
        if search_input.count() and search_input.first.is_visible():
            actions.append("search[<your query>]")

        for text in self._visible_texts("a.product-link"):
            actions.append(f"click[{text}]")
        for text in self._visible_texts("button"):
            if text.casefold() != "search":
                actions.append(f"click[{text}]")
        for text in self._visible_texts("label"):
            actions.append(f"click[{text}]")

        selected_options: dict[str, str] = {}
        checked = self._page.locator("input[type=radio]:checked")
        for index in range(checked.count()):
            item = checked.nth(index)
            selected_options[item.get_attribute("name") or f"option_{index}"] = item.get_attribute("value") or ""

        reward = None
        reward_locator = self._page.locator("#reward pre")
        if reward_locator.count():
            try:
                reward = float(reward_locator.first.inner_text().strip())
            except ValueError:
                reward = None

        return BrowserObservation(
            session_id=self._session_id,
            url=self._page.url,
            page_type=self._page_type(self._page.url),
            task=task,
            text=body_text,
            available_actions=tuple(self._deduplicate(actions)),
            selected_options=selected_options,
            reward=reward,
            done=self._page.locator("#thankyou").count() > 0,
        )

    def step(self, action: str, *, confirm_purchase: bool = False) -> ActionResult:
        self._ensure_started()
        self._logical_step += 1
        action_id = f"{self._session_id}-step-{self._logical_step}"
        before = self.observe()

        try:
            parsed = parse_action(action)
        except ValueError as exc:
            return self._finish(
                action_id, action, ExecutionStatus.INVALID_ACTION, before, error=str(exc)
            )

        legal_action = match_legal_action(parsed, before.available_actions)
        if legal_action is None:
            return self._finish(
                action_id,
                parsed.normalized,
                ExecutionStatus.INVALID_ACTION,
                before,
                error="Action is not present in the current legal-action set",
            )

        side_effect_key = self._side_effect_key(parsed, before)
        if parsed.has_side_effect:
            if self.config.require_purchase_confirmation and not confirm_purchase:
                return self._finish(
                    action_id,
                    parsed.normalized,
                    ExecutionStatus.CONFIRMATION_REQUIRED,
                    before,
                    error="Purchase requires explicit confirm_purchase=True",
                )
            if side_effect_key in self._side_effects:
                return self._finish(
                    action_id,
                    parsed.normalized,
                    ExecutionStatus.DUPLICATE_BLOCKED,
                    before,
                    error=f"Purchase action already {self._side_effects[side_effect_key]}",
                )

        retries = 0
        while True:
            injected = self.faults.take(self._logical_step)
            if injected == FaultKind.LOGIN_EXPIRED:
                self._recover_simulated_login(before.url, action_id)
                before = self.observe()
                continue
            if injected == FaultKind.RATE_LIMIT:
                self._show_status_overlay(
                    "HTTP 429 · 请求过于频繁",
                    "Browser Harness 正在按 Retry-After 退避，随后安全重试。",
                    "#b45309",
                    "429_rate_limit",
                )
                self.trace.write(
                    "rate_limited",
                    action_id=action_id,
                    action=parsed.normalized,
                    http_status=429,
                    retry_after=self.faults.retry_after_seconds,
                )
                if retries >= self.config.max_retries:
                    return self._finish(
                        action_id,
                        parsed.normalized,
                        ExecutionStatus.INFRA_FAILURE,
                        before,
                        retries=retries,
                        http_status=429,
                        infra_failure=True,
                        error="Rate-limit retry budget exhausted",
                    )
                self._sleep_before_retry(retries, self.faults.retry_after_seconds)
                retries += 1
                continue
            if injected == FaultKind.TIMEOUT:
                self._show_status_overlay(
                    "页面加载超时",
                    "Browser Harness 未执行重复副作用，正在恢复页面并重试。",
                    "#b91c1c",
                    "timeout_recovery",
                )
                self.trace.write(
                    "injected_timeout",
                    action_id=action_id,
                    action=parsed.normalized,
                    retry=retries,
                )
                if retries >= self.config.max_retries:
                    return self._finish(
                        action_id,
                        parsed.normalized,
                        ExecutionStatus.INFRA_FAILURE,
                        before,
                        retries=retries,
                        infra_failure=True,
                        error="Timeout retry budget exhausted",
                    )
                self._sleep_before_retry(retries)
                retries += 1
                continue

            if parsed.has_side_effect:
                self._side_effects[side_effect_key] = "pending"
            try:
                http_status = self._execute(parsed)
            except Exception as exc:
                # 不可逆动作提交后的结果可能处于未知状态，因此禁止自动重复点击。
                if parsed.has_side_effect:
                    return self._finish(
                        action_id,
                        parsed.normalized,
                        ExecutionStatus.UNCERTAIN_SIDE_EFFECT,
                        self.observe(),
                        retries=retries,
                        infra_failure=True,
                        error=str(exc),
                    )
                if retries >= self.config.max_retries:
                    return self._finish(
                        action_id,
                        parsed.normalized,
                        ExecutionStatus.INFRA_FAILURE,
                        self.observe(),
                        retries=retries,
                        infra_failure=True,
                        error=str(exc),
                    )
                self._page.goto(before.url, wait_until="domcontentloaded")
                self._sleep_before_retry(retries)
                retries += 1
                continue

            after = self.observe()
            self._capture_screenshot(
                f"{self._logical_step:02d}_{after.page_type}"
            )
            if parsed.has_side_effect:
                self._side_effects[side_effect_key] = "completed"
            return self._finish(
                action_id,
                parsed.normalized,
                ExecutionStatus.OK,
                after,
                retries=retries,
                http_status=http_status,
            )

    def _execute(self, action: ParsedAction) -> int | None:
        if action.kind == "search":
            self._page.locator("#search_input").fill(action.argument)
            locator = self._find_exact("button", "Search")
        else:
            locator = self._find_exact("a.product-link", action.argument)
            if locator is None:
                locator = self._find_exact("label", action.argument)
            if locator is None:
                locator = self._find_exact("button", action.argument)
            if locator is None:
                raise ValueError(f"No visible DOM element matches click[{action.argument}]")

        with self._page.expect_navigation(wait_until="domcontentloaded") as navigation:
            locator.click()
        response = navigation.value
        return response.status if response is not None else None

    def _recover_simulated_login(self, return_url: str, action_id: str) -> None:
        self.trace.write("login_expired", action_id=action_id, return_url=return_url)
        self._page.set_content(
            """
            <html><head><style>
              body{margin:0;background:#f4f6f8;color:#172033;
                   font-family:Segoe UI,Microsoft YaHei,sans-serif}
              .top{background:#131921;color:white;padding:18px 6%;
                   font-size:25px;font-weight:700}
              .shell{display:flex;align-items:center;justify-content:center;
                     min-height:620px}
              .card{width:430px;background:white;border:1px solid #d5d9d9;
                    border-radius:14px;padding:34px 38px;
                    box-shadow:0 18px 55px rgba(15,23,42,.13)}
              h1{margin:0 0 12px;color:#b45309}.notice{color:#64748b;
                 line-height:1.6;margin-bottom:22px}
              label{display:block;font-weight:600;margin:15px 0 6px}
              input{box-sizing:border-box;width:100%;padding:12px;
                    border:1px solid #94a3b8;border-radius:7px}
              button{width:100%;margin-top:23px;padding:13px;border:0;
                     border-radius:8px;background:#f59e0b;font-weight:700}
            </style></head><body>
              <div class="top">WebShop · Browser Harness</div>
              <main class="shell"><section class="card">
                <h1>登录状态已过期</h1>
                <div class="notice">会话失效，Harness将安全恢复登录，
                随后返回原商品页面继续执行。</div>
                <label>测试账号</label><input id="harness-user">
                <label>密码</label><input id="harness-password" type="password">
                <button id="harness-login">恢复会话</button>
              </section></main>
            </body></html>
            """
        )
        self._capture_screenshot(f"{self._logical_step:02d}_login_expired")
        # 模拟凭据只在Harness内部使用，不进入Agent Prompt或轨迹文件。
        self._page.locator("#harness-user").fill("harness-test-user")
        self._page.locator("#harness-password").fill("not-a-real-secret")
        if self.config.demo_event_pause_seconds:
            self._page.wait_for_timeout(self.config.demo_event_pause_seconds * 1000)
        self._page.locator("#harness-login").click()
        self._page.goto(return_url, wait_until="domcontentloaded")
        self.trace.write("login_recovered", action_id=action_id, return_url=return_url)

    def _show_status_overlay(
        self,
        title: str,
        detail: str,
        color: str,
        screenshot_name: str,
    ) -> None:
        if not self.config.demo_event_pause_seconds:
            return
        self._page.evaluate(
            """
            ([title, detail, color]) => {
              const existing = document.getElementById('harness-status-overlay');
              if (existing) existing.remove();
              const overlay = document.createElement('div');
              overlay.id = 'harness-status-overlay';
              overlay.style.cssText = [
                'position:fixed', 'inset:0', 'z-index:2147483647',
                'background:rgba(15,23,42,.78)', 'display:flex',
                'align-items:center', 'justify-content:center',
                'font-family:Segoe UI,Microsoft YaHei,sans-serif'
              ].join(';');
              overlay.innerHTML = [
                `<div style="background:white;border-top:8px solid ${color};`,
                'padding:32px 38px;border-radius:16px;max-width:620px;',
                'box-shadow:0 24px 70px rgba(0,0,0,.35)">',
                `<div style="font-size:28px;font-weight:700;color:${color};`,
                `margin-bottom:14px">${title}</div>`,
                '<div style="font-size:18px;line-height:1.7;color:#334155">',
                `${detail}</div></div>`
              ].join('');
              document.body.appendChild(overlay);
            }
            """,
            [title, detail, color],
        )
        self._capture_screenshot(
            f"{self._logical_step:02d}_{screenshot_name}"
        )
        self._page.wait_for_timeout(self.config.demo_event_pause_seconds * 1000)
        self._page.locator("#harness-status-overlay").evaluate("element => element.remove()")

    def _capture_screenshot(self, name: str) -> None:
        if self.config.screenshot_dir is None:
            return
        directory = Path(self.config.screenshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        images = self._page.locator("img")
        if images.count():
            try:
                self._page.wait_for_function(
                    "Array.from(document.images).every(image => "
                    "image.complete && image.naturalWidth > 0)",
                    timeout=15_000,
                )
            except Exception:
                # 单张第三方图片失效时仍需保留其余页面与轨迹证据。
                pass
        self._page.wait_for_timeout(750)
        self._page.screenshot(path=str(directory / f"{name}.png"), full_page=True)

    def _finish(
        self,
        action_id: str,
        action: str,
        status: ExecutionStatus,
        observation: BrowserObservation,
        *,
        retries: int = 0,
        http_status: int | None = None,
        infra_failure: bool = False,
        error: str | None = None,
    ) -> ActionResult:
        result = ActionResult(
            action_id=action_id,
            action=action,
            status=status,
            observation=observation,
            retries=retries,
            http_status=http_status,
            infra_failure=infra_failure,
            error=error,
        )
        self.trace.write("action_result", result=result.to_dict())
        return result

    def _route_request(self, route: Any) -> None:
        request_url = route.request.url
        base = urlparse(self.config.base_url)
        target = urlparse(request_url)
        same_origin = target.hostname == base.hostname and target.port == base.port
        if not same_origin or route.request.resource_type in {"image", "font"}:
            route.abort()
        else:
            route.continue_()

    def _find_exact(self, selector: str, text: str) -> Any | None:
        candidates = self._page.locator(selector)
        for index in range(candidates.count()):
            candidate = candidates.nth(index)
            if candidate.is_visible() and self._clean_text(candidate.inner_text()).casefold() == text.casefold():
                return candidate
        return None

    def _visible_texts(self, selector: str) -> list[str]:
        values: list[str] = []
        candidates = self._page.locator(selector)
        for index in range(candidates.count()):
            candidate = candidates.nth(index)
            if candidate.is_visible():
                text = self._clean_text(candidate.inner_text())
                if text:
                    values.append(text)
        return values

    def _sleep_before_retry(self, retry: int, minimum: float = 0.0) -> None:
        exponential = self.config.backoff_base_seconds * (2**retry)
        jitter = random.uniform(0, self.config.backoff_jitter_seconds)
        time.sleep(max(minimum, exponential) + jitter)

    def _side_effect_key(self, action: ParsedAction, observation: BrowserObservation) -> str:
        raw = f"{observation.session_id}|{observation.url}|{action.normalized.casefold()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result

    @staticmethod
    def _page_type(url: str) -> str:
        path = urlparse(url).path
        if path.startswith("/search_results/"):
            return "search_results"
        if path.startswith("/item_page/"):
            return "item"
        if path.startswith("/item_sub_page/"):
            return "item_subpage"
        if path.startswith("/done/"):
            return "done"
        return "search"

    def _ensure_started(self) -> None:
        if self._page is None:
            raise RuntimeError("Harness is not started; use it as a context manager or call start()")
