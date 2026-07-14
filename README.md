# Dynamic-Policy Content Risk Agent

## Same-day English SingGuard data pilot

The repository now includes a one-command, text-only pipeline for the first real
SingGuard-style dataset. It fixes eight-domain quotas before any model call, asks
Gemini to realize content, uses a separate blind Gemini pass to judge shuffled
before/after policies, applies deterministic local quality gates, and exports
accepted pairs directly to ms-swift `messages` JSONL.

Start with the credential-free plan check:

```bash
pip install -e '.[dev,teacher]'
python scripts/generate_singguard_data.py outputs/singguard-plan \
  --anchors 100 --seed 42 --plan-only
```

Then follow [the data runbook](docs/singguard-data-runbook.md) to fetch governed
style seeds and run the 100-anchor / 200-row pilot. Do not start the 2,000-anchor
batch until `quality_report.json` and the stratified human-review sample pass.

一个面向内容风控实验的最小仓库：先复现 SingGuard 最有价值的动态策略训练思路，再扩展到最多 3 轮工具调用。

当前代码只做三件事：

1. 构造动态 policy、反事实、SFT、GRPO 和 OPSD 数据；
2. 提供封闭世界的风控工具环境与确定性评测；
3. 用官方 ms-swift 命令训练 Qwen3-VL。

训练执行层不再自研 launcher，也不维护一套 YAML 配置框架。四个 shell 脚本就是完整训练入口，默认针对单机 8×H20。

## 当前状态

| 模块 | 状态 |
|---|---|
| 动态 policy 数据契约与规则反事实 | 已实现 |
| 公开数据导入与来源记录 | 已实现 |
| Gemini 生成两跳 SFT 候选 | 已实现 |
| SingGuard fast/slow 格式与动态 policy SFT bundle | 已实现 |
| Track A 单轮 SFT / GRPO / OPSD 数据 | 已实现 |
| Track B 最多 3 轮工具轨迹与环境 | 已实现 |
| Qwen3-VL 图像/视频字段透传 | 已实现 |
| 8×H20 官方 ms-swift 训练脚本 | 已实现，尚待服务器 smoke test |
| ms-swift 多轮 Agentic GRPO scheduler | 尚未实现 |

这不是 SingGuard 官方代码或完整复现。当前复现重点是动态 policy conditioning、策略反事实、Oracle 隔离和 SFT → RL 的实验闭环。

## 安装

开发与数据构造：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

训练服务器：

```bash
pip install -e '.[train]'
pip install 'vllm>=0.11.0'
```

Qwen3-VL 环境要求来自 ms-swift 官方最佳实践：`transformers>=4.57`、`qwen-vl-utils>=0.0.14`、`ms-swift>=4.0`。

## SingGuard SFT 主线

这条路径复现 SingGuard 的 policy-conditioned 分类格式，与后面的 Agentic JSON 工具调用路径相互独立。每条源数据都包含素材、非空的当前生效 policy、policy 变化类型、标注和 `fast` / `slow` 思考类型。相同 `split_group` 的策略反事实始终进入同一 split，避免同一素材泄漏到训练集和评测集。

先用最小 fixture 验证数据导出；它覆盖 `unsafe→unsafe`、`unsafe→safe`、`safe→unsafe`、`safe→safe` 四类 policy 变化：

```bash
rm -rf outputs/singguard-smoke

python scripts/prepare_singguard_sft.py \
  data/fixtures/singguard_examples.jsonl \
  outputs/singguard-smoke \
  --train-ratio 1 \
  --dev-ratio 0 \
  --holdout-ratio 0
```

输出的 `train.jsonl` 是 ms-swift 原生 `messages` 格式，可以直接训练：

```bash
TRAIN_DATA=outputs/singguard-smoke/train.jsonl \
VAL_DATA= \
OUTPUT_DIR=outputs/qwen3_vl_8b_singguard_sft \
bash scripts/train_qwen3_vl_sft.sh
```

`fast` 标签只输出安全结论和命中规则；`slow` 标签会按当前 policy 顺序逐条检查规则。unsafe 答案必须是当前生效规则的标题，safe 答案固定为 `Safe`，从数据契约层杜绝空 policy 和不可学习的规则标识。

## 最短跑通路径

### 1. 构造 SFT bundle

仓库自带一组纯合成 fixture，可先验证全流程：

```bash
python scripts/prepare_ms_swift_sft.py track_a \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/sft \
  --train-ratio 1 --dev-ratio 0 --holdout-ratio 0
```

输出包含 `train.jsonl`、`dev.jsonl`、`holdout.jsonl` 和一个只记录切分比例与数量的轻量 `manifest.json`。同一 `asset_id` 的不同 policy version 始终进入同一 split，防止素材泄漏。

### 2. SFT

```bash
TRAIN_DATA=outputs/sft/train.jsonl \
VAL_DATA= \
bash scripts/train_qwen3_vl_sft.sh
```

默认模型是 `Qwen/Qwen3-VL-8B-Instruct`，8 卡 LoRA，冻结视觉塔和 aligner。可通过 `MODEL`、`TRAIN_DATA`、`VAL_DATA`、`OUTPUT_DIR` 覆盖必要路径；其余参数直接修改脚本即可。

### 3. 构造 GRPO 或 OPSD bundle

```bash
python scripts/prepare_ms_swift_rl.py grpo \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/grpo \
  --train-ratio 1 --dev-ratio 0 --holdout-ratio 0

python scripts/prepare_ms_swift_rl.py opsd \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/opsd \
  --train-ratio 1 --dev-ratio 0 --holdout-ratio 0
```

GRPO 行格式：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."}],"solution":"{...}"}
```

`solution` 只供 reward 使用，不进入学生 prompt。OPSD 使用相同的 prompt-only `messages`，并增加只供 teacher 使用的 `teacher_prompt`。

### 4. GRPO

先在终端 1 启动 4 卡 rollout server：

```bash
bash scripts/start_qwen3_vl_rollout.sh
```

再在终端 2 用另外 4 卡训练：

```bash
SFT_ADAPTER=outputs/qwen3_vl_8b_sft/best \
TRAIN_DATA=outputs/grpo/train.jsonl \
VAL_DATA= \
bash scripts/train_qwen3_vl_grpo.sh
```

GRPO 使用三个确定性 reward：JSON 格式、label 和 rule，权重为 `0.05 / 0.75 / 0.20`。SFT adapter 同时作为 policy 与 reference adapter，符合 ms-swift 的 LoRA GRPO 用法。

### 5. OPSD

rollout server 保持运行，执行：

```bash
SFT_ADAPTER=outputs/qwen3_vl_8b_sft/best \
TRAIN_DATA=outputs/opsd/train.jsonl \
VAL_DATA= \
bash scripts/train_qwen3_vl_opsd.sh
```

OPSD 使用 `rlhf_type=gkd`、`lmbda=1`、`beta=1`。不设置 `teacher_model`：同一模型动态充当 student 和 teacher，teacher 额外看到 `teacher_prompt` 中的特权答案。

## 多模态数据

`Task` 可选字段：

```json
{
  "initial_observation": "<image>\n判断该广告是否违规。",
  "images": ["/data/ad.jpg"],
  "videos": []
}
```

占位符数量和媒体列表应一致。数据导出会原样保留 `images` / `videos`，交给 ms-swift 的 Qwen3-VL template 处理。

## 数据合成

### 公开数据

编辑 `configs/public_sources.example.yaml` 后运行：

```bash
python scripts/import_public_data.py \
  configs/public_sources.example.yaml \
  outputs/public_data
```

导入器负责格式统一、来源元数据与哈希，不替代人工确认数据许可。示例和约束见 `data/README.md`、`docs/public-data-card.md`。

### Gemini 两跳轨迹

安装 Gemini SDK：

```bash
pip install -e '.[teacher]'
```

鉴权只从环境变量读取，任选一种方式。Gemini Developer API：

```bash
export GEMINI_API_KEY=...
```

或者 Vertex AI：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=...
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=true
```

运行小批量合成：

```bash
python scripts/generate_teacher_sft.py \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  data/fixtures/cases.jsonl \
  data/fixtures/evidence.jsonl \
  outputs/teacher_candidates.jsonl \
  --model gemini-3.1-flash-lite \
  --allow-external-data \
  --max-records 100 \
  --max-requests 100
```

生成结果仍需经过本地封闭案例库、证据库和轨迹校验，模型不能自行创造 case ID；证据 ID 只在工具返回中作为内部定位字段，不属于最终判定目标。

`Task.images` 为空时只发送文字；非空时读取本地 JPEG、PNG、WebP 或 GIF，并与文字 prompt 一起发送。本地路径只用于读取文件，不会进入 Gemini prompt。当前 Gemini 合成不下载 HTTP 图片，也不处理视频。

## 核心数据边界

- `Task`：素材、当前 `policy_version`、完整 `active_policy`、最多 3 轮预算；
- `Oracle`：冻结标签、命中规则与处置，仅用于导出答案、reward 和评测；
- `CaseStore` / `EvidenceStore`：工具唯一可见的封闭底库；
- `RiskEnvironment`：`get_rule_detail`、`search_case`、`inspect_evidence`、`final_decision`；
- split key：`asset_id`，而不是单条样本；
- holdout：只评测，不参与 prompt、工具 observation 或 reward 输入。

规则反事实应保持素材不变，只切换 policy 条件和对应 Oracle。这样才能测出模型是否真的遵循动态策略，而不是记住素材标签。

## 下一步

1. 在 8×H20 上做 16～64 条数据的 SFT、GRPO、OPSD smoke test，记录显存、吞吐和首个 checkpoint；
2. 用真实公开图文素材扩充动态策略与反事实集；
3. 把 `RiskEnvironment` 接入 ms-swift `MultiTurnScheduler`，只支持现有 4 个工具和最多 3 轮；
4. 比较 Track A、Tool-SFT、Agentic GRPO 的 frozen holdout 指标。

停止门槛：若多轮 RL 只增加调用次数和延迟，却不能稳定改善误伤、漏放、证据完整度或动态 policy 一致性，就保留 SFT/Tool-SFT，不继续扩大 RL。

## 参考

- [Qwen3-VL ms-swift 最佳实践](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/Qwen3-VL-Best-Practice.md)
- [ms-swift GRPO](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md)
- [ms-swift GKD / OPSD](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GKD.md)
