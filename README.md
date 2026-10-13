# ShopPilot-RL：基于专家轨迹与强化学习的 Web 购物智能体

ShopPilot-RL 面向 WebShop 长程网页交互任务，基于 **Qwen2.5-3B-Instruct** 构建多轮购物智能体，将商品检索、候选比较、页面浏览、规格选择和购买决策建模为连续的 `Observation → Action → Reward` 过程。

项目采用“**专家轨迹行为冷启动 + LoRA 参数高效训练 + GRPO 在线策略优化**”的两阶段方案：先自主生成并校验购物轨迹，使模型掌握 WebShop 的基本操作闭环；再通过同一任务下多条 rollout 的相对奖励优化决策策略。所有关键实验均保存逐步页面、合法动作、模型响应、执行动作和环境奖励，支持完整的 bad-case 回溯。

## 项目亮点

- 自主构建 **5,000 条成功购物轨迹、28,625 个决策步骤**，覆盖搜索、商品点击、属性选择与最终购买。
- 额外构建 2,500 条搜索改写、翻页探索和候选比较课程轨迹，缓解首项偏好与无效循环。
- 实现 **Think/Action 分段加权 SFT**：Prompt 不参与损失，`Think` Token 权重为 `0.2`，关键 `Action` Token 权重为 `1.0`。
- 使用 LoRA 训练 Qwen2.5-3B-Instruct，并在单张 48 GB GPU 上完成 30 次 GRPO 参数更新。
- 在固定 128 个 WebShop 任务上，三轮 SFT 成功率分别为 **87.50%、86.72%、92.97%**；最终达到 **119/128**，平均任务得分 **0.9462**，动作合法率 **100%**。
- GRPO 采用 `B=8、G=8`，每次更新生成 64 条轨迹；第 10/20/30 步固定验证成功率均为 **93.75%**，并通过奖励方差、优势值、梯度范数和失败轨迹定位强化学习的有效信号与性能瓶颈。

> 指标口径：主结果来自 WebShop-small 固定任务评测，目标商品 ASIN 不强制与训练轨迹隔离，衡量固定商品库中新购物指令的完成能力。

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
| Epoch 3 | **119/128** | **92.97%** | **0.9462** | **100%** | **5.73** | **4.62%** |

### GRPO：固定 32 任务训练期验证

| 参数更新步 | 成功数 | 成功率 | 平均任务得分 | 合法动作率 | 平均轨迹长度 |
|---|---:|---:|---:|---:|---:|
| Step 10 | 30/32 | 93.75% | 0.9375 | 100% | 6.09 |
| Step 20 | 30/32 | 93.75% | 0.9375 | 100% | 6.25 |
| Step 30 | 30/32 | 93.75% | 0.9375 | 100% | 6.09 |

实验表明，专家轨迹 SFT 承担了主要能力冷启动；当策略起点已经较强时，同任务组内多条轨迹容易同时成功，GRPO 的相对优势信号随之减弱。项目因此建立了针对组内奖励标准差、有效任务组比例、梯度范数和逐步轨迹的联合诊断流程。

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
