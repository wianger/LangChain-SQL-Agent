# LangChain SQL Agent Benchmark 实验流程细节报告

本仓库的 Benchmark 旨在系统性地评估 LangChain SQL Agent 在多个复杂问答数据集（如 LoCoMo, SyllabusQA 等）上的真实表现。与单纯的静态指标评测不同，本框架模拟了从数据入库、并发检索到数据清理的完整生命周期，并在执行过程中全面追踪时间延迟和 Token 消耗。

以下将详细描述 LangChain SQL Agent 在评测框架中的实验流程细节。

## 文档入库与数据库初始化阶段

### SQLite 数据库初始化
为了支持高并发的读写操作并避免锁表问题，系统首先初始化一个本地 SQLite 数据库，并强制开启了 **WAL (Write-Ahead Logging)** 模式和较长的繁忙超时时间（`PRAGMA busy_timeout=10000`）。
随后，系统根据当前评测的数据集（如 LoCoMo 的对话数据或 SyllabusQA 的课程大纲）执行预定义的 DDL 语句，创建所需的结构化数据表。

### 数据解析化与导入
实验支持两种数据摄入模式：**Bulk（全量模式）** 和 **Per-Question（按需模式）**。

- **Bulk 模式（默认）**：在开始查询前，系统通过原生的 SQLAlchemy 批量执行 `INSERT` 语句，将整个数据集的结构化数据（如对话轮次、摘要、观察记录等）一次性写入 SQLite。该过程不调用 LLM，纯粹测试底层存储引擎的写入性能，并记录总体耗时。
- **Per-Question 模式**：为避免跨样本的数据污染或上下文干扰，系统会在针对每个样本组发起查询前，仅将当前相关的对话数据实时插入数据库，查询完成后立即清理。

## 代理查询与检索阶段

检索阶段是评测的核心，它测试了 LangChain SQL Agent 如何利用自然语言意图自主探索数据库并提取正确答案。

### 代理实例构建 (Agent Construction)
系统通过 `build_agent` 函数为评测初始化 LangChain 代理：
1. **工具包装配**：实例化 `SQLDatabaseToolkit`，它自动提供了用于获取表列表、查询表结构、检查 SQL 语法和执行 SQL 的核心工具。
2. **扩展工具**：由于原生 Toolkit 偏向只读，系统通过 `_make_write_tool` 注入了一个自定义的 `sql_db_write` 工具，赋予代理在需要时执行 `INSERT/UPDATE/DELETE` 的能力。
3. **引擎配置**：代理使用 `create_tool_calling_agent` 和 `AgentExecutor` 组装，这种结构化的 Tool Calling 方式比传统的文本正则解析（ReAct parsing）更加稳定可靠。
4. **Token 追踪**：将自定义的 `TokenTracker` 作为回调（Callback）注入到 LLM 中，以精确统计每次查询过程中的 Prompt Token 和 Completion Token 消耗。

### 并发检索执行 (Concurrent Retrieval)
为了加速评测，系统采用异步并发架构（基于 `asyncio` 和 Semaphore 信号量控制）。
- **Prompt 构建**：对于采样出的每一个 QA 样本，系统构建包含上下文约束（如 `sample_id` 和说话人信息）和具体问题的 Prompt。
- **Agent 执行**：调用 `arun_agent` 将问题提交给代理。代理在内部开启 ReAct 循环：
  1. 调用 `sql_db_list_tables` 观察可用表。
  2. 调用 `sql_db_schema` 获取目标表的 DDL。
  3. 生成 SQL 并调用 `sql_db_query` 执行查询。
  4. 综合查询结果生成最终的自然语言答案。
- **中间步骤提取**：系统不仅捕获代理的最终回答，还会解析 `AgentExecutor` 返回的 `intermediate_steps`。凡是成功执行的 `sql_db_query` 的输出（即实际从数据库中检索出的数据行）都会被收集起来，作为后续计算 Recall 的依据（`retrieved_texts`）。

## 数据清理阶段

### 状态重置 (Teardown)
- **Bulk 模式下**：在所有查询并发完成后，系统执行 `_bulk_delete` 函数，通过原生 SQL `DELETE FROM <table>` 清空所有数据表，恢复数据库的初始空状态，并记录删除操作的总耗时。
- **Per-Question 模式下**：针对特定样本组的查询一旦完成，系统立即执行特定 `sample_id` 的 `DELETE` 操作，确保下一个样本组的查询环境是干净的。

该清理过程不涉及 LLM 调用，因此 Token 消耗为零。

## 指标评估阶段

查询流程结束后，系统将收集到的预测答案（Predictions）、真实答案（Ground Truths）、真实证据（Evidences）以及代理实际检索到的中间数据（Retrieved Texts）统一送入指标计算模块。

系统主要评估以下几个维度：
1. **F1 Score**：计算预测答案与真实答案之间的 Token 级重叠度。
2. **Recall (检索召回率)**：评估代理通过 SQL 提取的中间结果（`retrieved_texts`）或最终答案是否有效覆盖了 Ground Truth 所在的证据文本（Evidence）。
3. **Accuracy (语义准确率)**：调用一个额外的裁判 LLM（`accuracy_llm`），让其判断代理的预测答案是否在语义上正确回答了原问题。
4. **性能与成本指标**：汇总 `OperationTracker` 记录的数据，输出检索阶段的平均耗时（`avg_time`）和平均 Token 消耗（`avg_tokens`）。

最终，所有的指标和每个样本的详细执行结果将按类别聚合，并序列化保存到 `results/` 目录下的 JSON 报告文件中。