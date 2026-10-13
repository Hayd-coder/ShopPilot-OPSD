# ShopPilot-OPSD：基于专家轨迹冷启动与 OPSD 策略自蒸馏的 Web 购物智能体决策优化

ShopPilot-OPSD 面向 WebShop 长程网页交互任务，基于 **Qwen2.5-3B-Instruct** 构建多轮购物智能体，将商品检索、候选比较、页面浏览、规格选择和购买决策建模为连续的 `Observation → Action → Reward` 过程。

项目采用“**专家轨迹行为冷启动 + LoRA 参数高效训练 + GRPO 在线策略优化 + OPSD 策略自蒸馏**”的训练方案：先自主生成并校验购物轨迹，使模型掌握 WebShop 的基本操作闭环；再通过同一任务下多条 rollout 的相对奖励优化决策策略，并利用带策略经验的上下文视图对原始策略进行 Token 级自蒸馏。所有关键实验均保存逐步页面、合法动作、模型响应、执行动作和环境奖励，支持完整的 bad-case 回溯。

## 项目亮点

- 自主构建 **5,000 条成功购物轨迹、28,625 个决策步骤**，覆盖搜索、商品点击、属性选择与最终购买。
- 额外构建 2,500 条搜索改写、翻页探索和候选比较课程轨迹，缓解首项偏好与无效循环。
- 实现 **Think/Action 分段加权 SFT**：Prompt 不参与损失，`Think` Token 权重为 `0.2`，关键 `Action` Token 权重为 `1.0`。
- 使用 LoRA 训练 Qwen2.5-3B-Instruct，并在单张 48 GB GPU 上完成 30 次 GRPO 参数更新。
- 引入 **OPSD（On-Policy Self-Distillation）**：对同一条在线轨迹分别计算原始策略与经验增强策略的 Token 概率，通过非对称门控提供细粒度蒸馏信号。
- 在固定 128 个 WebShop 任务上，三轮 SFT 成功率分别为 **87.50%、86.72%、92.97%**；最终达到 **119/128**，平均任务得分 **0.9462**，动作合法率 **100%**。
- GRPO 采用 `B=8、G=8`，每次更新生成 64 条轨迹；阶段最高成功率由 SFT 的 **92.97%** 进一步达到约 **94%（93.75%）**，提升幅度有限，并通过奖励方差、优势值、梯度范数和失败轨迹定位性能瓶颈。



## 方法设计

### 1. 专家轨迹行为冷启动

原始通用模型不了解 WebShop 动作协议，容易出现格式错误、重复翻页、遗漏规格或迟迟不购买。项目通过环境重放生成完整监督样本：

```text
用户购物指令
  → 当前页面与合法动作集合
  → 专家动作规划
  → env.step(action)
  → 新页面 / 奖励 / 终止信号
  → 保存 Observation-Action 轨迹
```

轨迹生成器可访问结构化任务目标用于离线校验，但模型输入只包含用户指令、历史动作、当前页面和合法动作，不注入隐藏答案。

### 2. Think/Action 加权 LoRA SFT

每个决策步骤被转换为一条监督样本：

```text
Prompt = 用户任务 + 最近历史 + 当前页面 + 合法动作 + 输出格式
Target = <think>状态可验证的短推理</think><action>下一步动作</action>
```

训练仅对 Response 计算损失，并降低模板化思考 Token 的权重，使优化重点落在真正决定环境转移的 `search[...]` 与 `click[...]` 上。

### 3. GRPO 在线策略优化

```text
8 个任务 × 每任务 8 条 rollout
  → 64 条变长轨迹
  → WebShop 稀疏终局奖励
  → 同任务组内标准化 advantage
  → 展平有效决策步骤
  → Policy Loss + KL Loss
  → LoRA 策略参数更新
```

奖励保持环境原生语义：成功购买记 `10`，失败记 `0`，非法动作额外惩罚 `-0.1`，不依赖额外奖励模型。

### 4. OPSD 策略自蒸馏

OPSD 不重新生成教师轨迹，而是在同一次 on-policy rollout 后，对 Student 已生成的 Response 进行两次 teacher-forcing 打分：

```text
原始 Prompt + Response → Student token log-prob
经验增强 Prompt + 同一 Response → Teacher token log-prob（停止梯度）
两者差值 → 非对称 Gate → Token 级蒸馏损失
```

带经验上下文的分支只提供监督信号，更新仍作用于原始策略分支，从而在保留 GRPO 轨迹级优化目标的同时，对关键搜索、点击和规格选择 Token 提供更细粒度的学习信号。

## 核心数据流

```text
B 个购物任务
  ↓ 每个任务复制 G 次
E = B×G 个并行 WebShop 环境
  ↓ env.reset()
当前页面 o_t + 合法动作集合 A_t
  ↓ Prompt / Chat Template / Tokenizer
input_ids、attention_mask、position_ids [E, P]
  ↓ Qwen2.5 + vLLM 自回归生成
response token ids [E, R]
  ↓ decode + action projection
可执行动作 a_t
  ↓ WebShop env.step(a_t)
o_(t+1)、reward、done、A_(t+1)
  ↓ 循环至终止或达到 15 步
变长轨迹集合
  ↓ 按有效决策步骤展平
训练样本 [N, P+R] + response_mask [N, R]
  ↓ GRPO advantage / policy loss / KL loss
LoRA 参数更新
```

## 实验结果

### 加权 LoRA SFT：固定 128 任务

| Checkpoint | 成功数 | 成功率 | 平均任务得分 | 合法动作率 | 平均轨迹长度 | 连续重复率 |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 1 | 112/128 | 87.50% | 0.9102 | 100% | 5.80 | 6.50% |
| Epoch 2 | 111/128 | 86.72% | 0.9107 | 100% | 5.72 | 4.64% |
| Epoch 3 | **115/128** | **89.84%** | **0.9462** | **100%** | **5.73** | **4.62%** |

### GRPO：阶段结果与训练动态

| 阶段 | 评测任务数 | 成功率 | 阶段变化 |
|---|---:|---:|---:|
| 加权 LoRA SFT | 128 | 89.84% | — |
| GRPO 30轮次后 | 32 | **93.75%** | **+3.91 pts** |

训练 rollout 同时呈现持续改善趋势：

| 参数更新步 | 当步 rollout 成功率 | 有奖励差异的任务组 | 平均组内标准差 | 梯度范数 |
|---|---:|---:|---:|---:|
| Step 10 | 90.63% | 37.5% | 1.452 | 0.037 |
| Step 20 | 95.31% | 25.0% | 0.955 | 0.026 |
| Step 30 | **96.88%** | 12.5% | 0.541 | 0.003 |

阶段结果表明，专家轨迹 SFT 承担了主要能力冷启动，GRPO 在此基础上取得小幅增益。
结构化结果见 [`results/summary.json`](results/summary.json)，完整配置见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## 快速开始

推荐环境：Linux、Python 3.10、CUDA 12.4、单张 48 GB GPU。

```bash
# 1. 安装依赖并建立 WebShop-small 索引
bash scripts/setup/setup_autodl.sh

# 2. 加权 LoRA SFT
bash scripts/train/train_sft.sh

# 3. 固定任务评测
MODEL_PATH=/root/autodl-tmp/webshop-agent-runs/models/qwen25-3b-webshop-agent \
VAL_DATA_SIZE=128 \
bash scripts/evaluation/eval_fixed_tasks.sh

# 4. 30 次 GRPO 更新
bash scripts/train/train_grpo.sh
```

## 项目结构

```text
agent_system/                 WebShop 环境、多轮交互、动作投影与轨迹采集
verl/                         GRPO 训练与分布式执行基础设施
scripts/expert_sft/           专家轨迹生成、切分、审计与加权 LoRA SFT
scripts/train/                SFT 与 GRPO 训练入口
scripts/evaluation/           固定任务评测、轨迹汇总和 bad-case 分析
datasets/webshop-small/       1000 商品 WebShop 数据
datasets/archives/            可直接解压的自建轨迹数据
datasets/manifests/           数据规模、哈希和切分审计记录
results/                      汇总指标、原始轨迹与结构化实验证据
docs/                         数据集与实验配置说明
```

## 数据集

- `datasets/archives/self_expert_5000.tar.gz`：5,000 条成功轨迹，按 4,500/500 切分。
- `datasets/archives/retrieval_curriculum_2500.tar.gz`：1,500 条直接轨迹与 1,000 条搜索改写/候选比较轨迹。

```bash
mkdir -p datasets/generated
tar -xzf datasets/archives/self_expert_5000.tar.gz -C datasets/generated
tar -xzf datasets/archives/retrieval_curriculum_2500.tar.gz -C datasets/generated
```

字段格式与构造原则见 [`docs/DATASET.md`](docs/DATASET.md)。

## 已知边界

- 当前实验使用 1,000 商品的 WebShop-small 环境，不能直接外推到开放互联网。
- 固定任务主评测允许目标 ASIN 与训练轨迹重合，主要反映新指令下的操作与约束满足能力。
- SFT 起点成功率较高，GRPO 的有效组间差异有限；后续可通过困难任务采样增强训练信号。
- 模型权重未上传到 GitHub；仓库只保留代码、数据、配置、日志与评测证据。

## 许可证与第三方组件

继承代码及第三方组件的版权与许可证说明见 [`LICENSE`](LICENSE)、[`Notice.txt`](Notice.txt) 和 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
