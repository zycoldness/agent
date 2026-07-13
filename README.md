# Dynamic Policy Risk Agent

面向内容风控的动态策略训练与 2～3 轮受限 Agent 研究底座。项目先复现
[SingGuard](https://github.com/inclusionAI/SingGuard) 的核心研究问题——模型是否会依据**当前完整生效策略**改变裁决——再扩展到需要查询规则、案例或素材证据的多轮工具调用。

> [!IMPORTANT]
> 本项目目前是 **SingGuard-style 简化训练复现**，不是官方训练过程的逐项精确复现。
> 本仓库目前不包含 SingGuard 官方完整训练数据、训练脚本和全部超参数；这里实现的是动态 policy 数据契约、反事实数据、SFT 数据导出、确定性评测和受限 Agent 环境。已经提供基于 [ms-swift](https://github.com/modelscope/ms-swift) 4.3 的单卡 LoRA SFT 基线；官方模型 inference 适配、GRPO 和 OPD 仍在路线图中。

## 两条实验主线

```text
同一素材 + 完整 active policy
              │
              ├── Track A：无工具 Guard
              │      直接输出 final_decision
              │      验证动态策略跟随能力
              │
              └── Track B：闭世界 Agent
                     最多 2～3 个 assistant turn
                     查规则 / 查案例 / 查证据 → final_decision
                     只用于必须补充证据的 hard case
```

- **Track A** 是 SingGuard-style 主基线：模型每次收到完整 `active_policy` 和素材，直接输出结构化裁决。
- **Track B** 是本项目的 Agentic RL 扩展，并非 SingGuard 原始方法：最多 3 轮，只能调用 `get_rule_detail`、`search_case`、`inspect_evidence` 和 `final_decision`。
- 环境是闭世界的：规则查询只能命中当前生效规则；案例只含脱敏事实文本；证据只能属于当前素材；Oracle 只用于离线监督和终局打分。

## 当前实现状态

| 能力 | 状态 | 位置 |
|---|---|---|
| 不可变数据契约：Task、Policy、Evidence、Oracle、Action、Decision | 已实现 | `src/risk_agent/contracts.py` |
| 完整 active policy 渲染 | 已实现 | `src/risk_agent/policy.py` |
| 脱敏案例库与素材级证据库 | 已实现 | `src/risk_agent/stores.py` |
| 最多 3 轮的闭世界风控环境与确定性奖励 | 已实现 | `src/risk_agent/environment.py` |
| 规则新增、删除、改写、豁免反事实对 | 已实现 | `src/risk_agent/counterfactuals.py` |
| label、policy-following、rule、evidence 评测 | 已实现 | `src/risk_agent/evaluator.py` |
| 受控公开来源采集与原子落盘 | 已实现 | `src/risk_agent/crawler.py` |
| MM-SafetyBench / 监管案例公开素材规范化与审计清单 | 已实现 | `src/risk_agent/public_data.py`、`docs/public-data-card.md` |
| Track A / Track B messages JSONL 导出 | 已实现 | `src/risk_agent/sft_export.py` |
| 有预算、可审计的 Gemini teacher 候选生成 | 已实现 | `src/risk_agent/teacher.py`、`synthesis.py` |
| 官方 SingGuard 模型 inference / fast、fast-slow、slow 适配 | **尚未实现** | 路线图 |
| ms-swift 4.3 单卡 H20 SFT 数据切分、LoRA 配置和安全启动脚本 | 已实现 | `src/risk_agent/ms_swift_sft.py`、`ms_swift_launcher.py` |
| ms-swift 多轮 GYM / GRPO rollout 适配 | **尚未实现** | 路线图 |
| OPD 训练配置与教师/学生实验 | **尚未实现** | 路线图 |

## 安装

需要 Python 3.11 或更高版本。建议在独立虚拟环境中安装。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

只有需要调用 Gemini teacher 时才安装额外依赖：

```bash
python -m pip install -e ".[dev,teacher]"
```

服务器运行 SFT 时安装锁定在 4.3 minor 的训练依赖：

```bash
python -m pip install -e ".[dev,train]"
```

本地开发和单测不需要安装 `ms-swift`；只有非 `--dry-run` 启动才会检查 `ms-swift>=4.3,<4.4` 和 `swift` 可执行文件。GRPO 与 OPD 尚未实现。

## 数据边界

`data/fixtures/` 中只有手工构造的合成开发样例：同一条“保证七天减重”的 OCR，在包含或移除功效承诺规则时分别标为 `unsafe` 和 `safe`。它们用于验证协议、反事实和导出链路，不代表真实业务分布，也不能用于报告线上效果。

数据分层应保持为：

```text
公开原始来源（隔离） → 条款审查 / 去标识化 / 人工审核 → processed
合成 fixtures        → 单测、格式预热、链路调试
授权业务数据          → 单独审批、去标识化、按原始素材切分 → train / holdout
Oracle               → 只进终局监督和评测，不进检索或工具 observation
```

禁止将用户内容、商家信息、内部策略决策或未经授权的业务样本提交到公开仓库或外部 teacher。生产验证必须使用另行审批、去标识化的 holdout，并按原始素材切分，避免同一素材的不同 policy 版本跨训练集与评测集。

更完整的数据治理约束见 `data/README.md` 和 `docs/data-card.md`。

## 导出 SFT 数据

导出结果是通用 `messages` JSONL。单文件导出适合检查格式；真正训练前应继续构造分组切分 bundle。

### Track A：无工具 Guard

下面的命令可直接使用仓库内合成 fixture：

```bash
python scripts/export_sft.py \
  track_a \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/track_a_sft.jsonl
```

每条记录包含完整 active policy、素材输入和一个 `final_decision`。任务与 Oracle 必须按 `(asset_id, policy_version)` 一一对应；Oracle 引用的规则必须位于该任务的 active policy 中。

### Track B：工具轨迹

Track B 的第一个输入不是普通 Task JSONL，而是轨迹 JSONL。每行必须只有 `task` 和 `steps`；每个 step 的 `action` 与 `observation` 都是 JSON 字符串。最多允许 `task.max_turns - 1` 个工具 step，最后一个 `final_decision` 由 Oracle 生成。

轨迹行结构示例：

```json
{"task":{"asset_id":"synthetic-asset-1","policy_version":"synthetic-with-efficacy-rule","active_policy":[{"rule_id":"AD-001","title":"Absolute efficacy claim","text":"Do not guarantee a weight-loss result.","exceptions":[],"priority":100}],"initial_observation":"OCR: Guaranteed to lose ten pounds in seven days.","max_turns":3},"steps":[{"action":"{\"tool\":\"inspect_evidence\",\"arguments\":{\"kinds\":[\"ocr\"]}}","observation":"[{\"evidence_id\":\"ocr-1\",\"asset_id\":\"synthetic-asset-1\",\"kind\":\"ocr\",\"content\":\"Guaranteed to lose ten pounds in seven days.\"}]"}]}
```

导出时必须同时提供脱敏案例库和证据库：

```bash
python scripts/export_sft.py \
  track_b \
  data/processed/track_b_trajectories.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/track_b_sft.jsonl \
  --cases data/fixtures/cases.jsonl \
  --evidence data/fixtures/evidence.jsonl
```

`data/processed/track_b_trajectories.jsonl` 是待生成的输入路径，仓库当前没有预置该文件。导出器会重新执行确定性查询并核对 observation：案例只能含 `case_id/text`，证据只能含 `evidence_id/asset_id/kind/content`，自由文本、Oracle 字段、跨素材证据和伪造检索结果都会被拒绝。所有记录先完成校验，再原子写入输出文件。

### 构造 ms-swift train/dev/holdout bundle

Track A 示例：

```bash
python scripts/prepare_ms_swift_sft.py \
  track_a \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/track_a_bundle \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --holdout-ratio 0.1 \
  --seed 42
```

切分以 `asset_id` 为组：同一素材的不同 policy 版本和反事实变体不会跨 train/dev/holdout。小数据允许 dev 或 holdout 为空，`manifest.json` 会明确记录每个 split 的样本数与素材组数。三份训练文件每行只有标准 `messages`，不会附带 `asset_id`、Oracle 或分组元数据；这些信息只参与切分和最终监督。输出目录必须不存在；发布器先独占创建目录，最后才写入 complete manifest。若中途失败，会保留没有 manifest 的不完整目录供审计，需要人工确认后清理。

Track B 使用同一命令并增加闭世界底库：

```bash
python scripts/prepare_ms_swift_sft.py \
  track_b \
  data/processed/track_b_trajectories.jsonl \
  data/fixtures/oracle.jsonl \
  outputs/track_b_bundle \
  --cases data/fixtures/cases.jsonl \
  --evidence data/fixtures/evidence.jsonl \
  --seed 42
```

### dry-run 与单卡 SFT

先在不导入 ms-swift、不访问 GPU 的情况下验证 bundle 哈希和最终配置：

```bash
python scripts/launch_ms_swift_sft.py \
  outputs/track_a_bundle \
  --config configs/ms_swift/sft_qwen3_1_7b_lora.yaml \
  --output-dir outputs/runs/qwen3-1.7b-track-a \
  --device 0 \
  --dry-run
```

dry-run 输出完整 `rendered_config`、argv 和环境变量。若 dev 为空，会显式渲染 `val_dataset: []` 与 `eval_strategy: no`，不会拿 train 充当验证集。确认后移除 `--dry-run` 即可真实启动：

```bash
python scripts/launch_ms_swift_sft.py \
  outputs/track_a_bundle \
  --config configs/ms_swift/sft_qwen3_1_7b_lora.yaml \
  --output-dir outputs/runs/qwen3-1.7b-track-a \
  --device 0
```

基线为 `Qwen/Qwen3-1.7B`、`tuner_type: lora`、bf16、`max_length: 4096`、batch size 1 加梯度累积和 gradient checkpointing；默认不启用 DeepSpeed、vLLM 或 4-bit。真实启动会重新读取并验证 bundle，把已经验明哈希和 messages schema 的 train/dev 精确字节写入私有临时快照，再用官方 argv 形式 `swift sft <rendered.yaml>` 启动；YAML 只引用快照，不再引用可被并发替换的原文件。launcher 固定单进程单卡、不使用 shell，并在进程结束后清理快照和临时配置。可用 `--model /path/to/local/model` 覆盖模型。

## Gemini teacher 合成

Gemini teacher 用于生成“直接裁决”或“一次本地工具查询 + 最终裁决”的候选轨迹。Teacher 可以看到 Task 与 Oracle；如果选择查询，还会在第二次请求中看到本地重新计算的指定规则、案例或证据。返回动作必须与 Oracle 一致，最终学生 messages 由本地代码重建，teacher 自带的 observation 不会直接写入训练数据。

> [!CAUTION]
> 这是外部数据传输。脚本只允许 `synthetic` 或 `public` 分类，并强制要求显式传入 `--allow-external-data`。不要向 Gemini 发送真实用户内容、内部规则、业务 Oracle 或任何未获得外发授权的数据。

安装 teacher 依赖并设置以下任一环境变量：

```bash
export GEMINI_API_KEY="..."
# 或：export GOOGLE_API_KEY="..."
```

使用合成 fixture 的受限调用示例：

```bash
python scripts/generate_teacher_sft.py \
  data/fixtures/tasks.jsonl \
  data/fixtures/oracle.jsonl \
  data/fixtures/cases.jsonl \
  data/fixtures/evidence.jsonl \
  outputs/gemini_teacher_sft.jsonl \
  --model gemini-2.5-pro \
  --data-classification synthetic \
  --allow-external-data \
  --max-records 10 \
  --max-requests 20 \
  --max-attempts 3 \
  --max-output-tokens 2048
```

成本控制有两层：

- `--max-records` 限制成功记录数，`--max-requests` 限制包含重试在内的 provider 请求数；
- 若设置 `--max-estimated-cost`，还必须同时提供正数的 `--input-cost-per-million` 与 `--output-cost-per-million`。价格需按调用当日的官方定价填写；代码会按 UTF-8 prompt 字节数、固定 4096 token 余量和 `max_output_tokens` 做最坏情况预留，在发请求前硬性拒绝可能越过预算的调用。

带成本上限的参数形式如下：

```bash
  --max-estimated-cost "$MAX_COST_USD" \
  --input-cost-per-million "$INPUT_PRICE_USD_PER_M" \
  --output-cost-per-million "$OUTPUT_PRICE_USD_PER_M"
```

`configs/gemini_teacher.example.yaml` 只是 CLI 参数参考，**不会自动加载**。每次调用仍需显式传入 `--allow-external-data`。

输出采用 fail-closed 语义：

- 完成全部输入后写入最终 `.jsonl`，并保留状态为 `complete` 的 `.checkpoint.json`；
- 遇到记录数、请求数、成本上限、provider 失败或非法响应时，不产生最终文件；已完成行写入 `.partial`，原因与用量写入 `.checkpoint.json`；
- 输出、`.partial` 和 `.checkpoint.json` 在启动前都必须不存在；当前没有自动断点续跑，需先审计 partial/checkpoint，再选择新的输出路径重新运行或自行构造后续批次。

测试全部使用 fake teacher。本项目开发与 README 校验过程中**没有实际调用 Gemini**。

## 受控公开来源采集

先复制并修改 `configs/sources.example.yaml`。示例中的域名和 URL 只是占位符，每个真实 URL 都必须先确认使用条款与授权范围。

```bash
python scripts/fetch_sources.py configs/sources.example.yaml \
  --output-dir data/raw
```

采集器不是通用爬虫：只访问 manifest 中逐条列出的 HTTPS URL，要求精确域名 allowlist，遵守 `robots.txt`，不跟随重定向，并拒绝账号路径、凭据参数、本机/IP 地址、超大响应和符号链接输出路径。

每个完整来源包含：

```text
data/raw/<source-hash>.raw
data/raw/metadata/<source-hash>.json
data/raw/metadata/<source-hash>.complete.json
```

下游必须用 `risk_agent.crawler.is_complete_artifact(...)` 校验完成标记和哈希。下载结果仍处于隔离区，不能未经条款审查、去标识化、结构化转换和人工批准就进入 SFT/RL 数据。

要先用公开数据跑通素材链路，可复制 `configs/public_sources.example.yaml`，手工准备 MM-SafetyBench 本地 checkout/图片，并将监管页面先交给上面的受控 crawler：

```bash
python scripts/import_public_data.py \
  configs/public_sources.local.yaml \
  data/processed/public_seed
```

该命令不访问网络，按单文件/总字节预算读取，独占创建目标目录，并最后发布 `import_manifest.json` 完成标记；目标目录必须不存在。POSIX 读取和发布使用逐组件目录句柄与 `O_NOFOLLOW`，失败的 reservation 不按路径自动删除，需人工审计清理；Windows 仅提供 identity 变化时 fail-closed 的较弱保证。配置采用 schema v2，`local_html` 与 `crawler_artifact` 字段互斥，所有来源会在全局记录上限生效前完成校验，报告会列出被截断或跳过的来源（包括 MM-SafetyBench 自身达到上限且仍有记录时）。MM-SafetyBench 每条记录同时保留 CC BY-NC 4.0 非商业限制和上游 GPT-4/Stable Diffusion 限制。HTML 清理不等于 PII 脱敏；监管案例只有在 license、条款和内容审查全部获批后才可 `published`。Crawler 导入只信任完成哈希覆盖的治理 metadata，配置不能事后提升旧 artifact。这批公开数据只用于 pipeline/OOD 检查，不生成 `Task` 或 `Oracle`，也不等于 SingGuard 官方训练集。完整说明见 `docs/public-data-card.md`。

## 测试

```bash
python -m pytest -q
```

当前基线为：

```text
266 passed, 8 skipped
```

8 个 skip 来自 Windows 环境缺少稳定的符号链接权限、POSIX FIFO API 或 `dir_fd/openat` 语义，对应 crawler 与输出发布的 symlink/FIFO/路径替换防护测试；在支持相应能力的平台上会执行。测试不访问 Gemini，也不运行真实训练。

## 30 / 60 / 90 天继续或停止门槛

这些门槛用于避免把训练是否成功变成主观判断：

| 时间点 | 必须回答的问题 | 继续条件 | 停止/收缩条件 |
|---|---|---|---|
| 30 天 | 无工具 Guard 是否真的理解动态 policy？ | Track A 在规则新增、删除、改写、豁免四类反事实上通过预设 policy-following 门槛，并完成数据泄漏检查 | 未超过固定策略/多数类基线：暂停 RL，先修数据与策略表达 |
| 60 天 | 工具调用是否带来业务增益？ | Tool-SFT 在“必须依赖外部证据”的 hard slice 上显著超过 Track A slow，并保持越权率、格式错误率和延迟在预算内 | 未超过 Track A slow：保留无工具 Guard，停止扩大 Agent 训练 |
| 90 天 | Agentic RL 是否优于监督轨迹？ | GRPO 在冻结 holdout 上超过 Tool-SFT，且收益不是来自更多无效调用或数据泄漏 | 未超过 Tool-SFT：停止 RL，交付 Tool-SFT 或 Track A |

具体阈值应在查看 holdout 结果前冻结，并同时报告 label accuracy、policy-following accuracy、rule/evidence exact match、工具调用次数、非法调用率和延迟。

## 服务器使用

```bash
git clone https://github.com/zycoldness/agent.git
cd agent
git checkout feature/singguard-risk-agent

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python -m pytest -q
```

先用较小样本 bundle 做 H20 smoke run，建议保持默认 4096 长度，只跑 1 epoch，并观察显存、首个 logging step、checkpoint 写入和 dev 行为。准备 bundle、执行 dry-run 和真实启动的完整命令见上面的“导出 SFT 数据”。本地测试不会下载模型、访问 GPU 或实际训练。

## 路线图

1. 接入 SingGuard 官方模型，复用官方 inference，增加批量适配和 `fast / fast-slow / slow` 对比评测。
2. 增加 Track A 的 GRPO 与 OPD 配方，奖励以 label、active rule、evidence、policy counterfactual 为确定性信号。
3. 将现有 `RiskEnvironment` 适配到 ms-swift 多轮 rollout，先做 Tool-SFT，再做最多 3 轮 Agentic GRPO。
4. 使用获批、去标识化业务 holdout 执行 30/60/90 gate，并与无工具 slow Guard 比较效果、成本和延迟。

## 参考

- [SingGuard 官方仓库](https://github.com/inclusionAI/SingGuard)
- [ms-swift 官方仓库](https://github.com/modelscope/ms-swift)
- `docs/superpowers/specs/2026-07-13-singguard-multiturn-risk-agent-design.md`
- `docs/superpowers/plans/2026-07-13-singguard-multiturn-risk-agent.md`
