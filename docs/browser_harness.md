# Browser Harness

Browser Harness通过Playwright把ShopPilot现有的`search[...]`与`click[...]`动作协议映射为真实浏览器操作，让Agent在本地WebShop页面中完成搜索、候选比较、规格选择和购买。

## 系统位置

```text
用户任务
  → Agent生成Think与Action
  → Browser Harness解析并校验动作
  → Playwright控制Edge/Chromium执行
  → DOM页面更新
  → 提取Observation与合法动作
  → Agent继续下一轮决策
```

Agent负责决定搜索词、候选商品和目标规格；Harness负责动作执行、页面感知、异常恢复与安全控制。

## 核心能力

- 从DOM提取当前任务、可见页面、已选规格和合法动作。
- 校验`search[...]`、商品链接、规格标签与页面按钮。
- 按照`Retry-After`结合指数退避处理HTTP 429。
- 模拟Session过期，恢复登录后返回原页面继续执行。
- 对无副作用动作进行有限超时重试。
- 对购买动作执行显式确认、动作指纹和幂等保护。
- 将动作、异常、恢复、页面与奖励保存为脱敏JSONL Trace。
- 自动等待商品图片加载并生成关键步骤截图。

## 安装

```powershell
python -m pip install -r requirements-browser.txt
```

Windows可以直接复用本机Edge：

```powershell
python scripts/browser_harness/run_webshop_browser_smoke.py --channel msedge
```

Linux首次使用时安装Chromium：

```bash
python -m playwright install chromium
```

## 真实数据可视化演示

```powershell
python scripts/browser_harness/run_real_data_demo.py
```

脚本读取WebShop的1000商品快照和人工购物任务，自动执行：

```text
搜索任务
→ 429退避恢复
→ 8候选商品结果页
→ 登录状态恢复
→ 点击目标商品
→ 超时恢复
→ 选择目标规格
→ 确认购买
→ Reward=1.0
```

生成内容：

```text
results/browser_harness/real_data_demo.jsonl
results/browser_harness/screenshots/*.png
```

## 故障注入

故障按照Agent逻辑步骤配置，并且同一故障只触发一次：

```python
FaultPlan.from_spec("rate_limit=1;login=2;timeout=3")
```

含义：

- 第1步注入HTTP 429并退避重试。
- 第2步模拟登录状态过期并恢复Session。
- 第3步模拟页面超时并恢复原页面。

默认退避策略：

```text
delay = max(Retry-After, base × 2^retry) + jitter
```

## 购买安全

`click[buy now]`默认需要显式传入`confirm_purchase=True`。Harness根据`Session + URL + Action`生成动作指纹，并维护`pending/completed`状态。

购买请求提交后如果结果不确定，返回`uncertain_side_effect`并停止自动重试；相同购买动作再次提交时返回`duplicate_blocked`。

## Trace字段

每个Reset、异常、恢复和动作结果都会写入JSONL，核心字段包括：

- `action_id`、`action`、`status`
- `http_status`、`retries`、`infra_failure`
- `url`、`page_type`、`available_actions`
- `selected_options`、`reward`、`done`

基础设施异常通过`infra_failure`与商品误选、规格错误等Agent任务失败分开统计。

## 测试

```powershell
python -m pytest tests/browser_harness -q
```

测试覆盖动作解析、合法动作校验、一次性故障注入、真实Edge导航、429恢复、登录恢复、购买确认和终局奖励读取。
