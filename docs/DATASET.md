# 专家轨迹数据说明

## 数据来源与构造

轨迹由 `scripts/expert_sft/generate_oracle_webshop.py` 在 WebShop 环境中逐步执行并记录。生成器可以访问结构化任务目标，用于离线选择正确商品和验证终局结果；模型 Prompt 只包含用户任务、历史动作、当前页面和合法动作，不包含隐藏 ASIN 或隐藏商品标题。

| 字段 | 含义 |
|---|---|
| `trajectory_id` / `turn` | 轨迹编号与步骤编号 |
| `task` / `prompt` | 用户约束与模型输入 |
| `generated_think` | 基于可见状态的短推理 |
| `expert_action` | `search[...]` 或 `click[...]` |
| `observation_before/after` | 动作前后的页面文本 |
| `admissible_actions_before` | 当前合法动作集合 |
| `target_asin/options` | 离线审计使用的目标商品与规格 |
| `trajectory_mode` | 直接购买、搜索改写或候选比较 |
| `official_terminal_score` | WebShop 原生终局得分 |

## 数据规模

- 直接购物数据：5,000 条成功轨迹、28,625 个决策步骤，按 4,500/500 切分。
- 检索课程数据：2,500 条轨迹、16,337 个决策步骤，其中含 1,000 条搜索改写与候选比较轨迹。
- 完整统计、切分和 SHA256 位于 `datasets/manifests/`。

训练集和验证集按 `trajectory_id` 隔离；同一个 ASIN 可以对应不同购物指令和规格约束。主评测采用固定任务索引与确定性解码，衡量固定商品库中的购物任务完成能力。

## Think/Action 监督

```text
Think Token 权重  = 0.2
Action Token 权重 = 1.0
Prompt Token 权重 = 0.0
Padding 权重      = 0.0
```

`generated_think` 必须由任务、当前页面、合法动作和历史记录支持，不能因为已知专家动作而编造不可见信息。
