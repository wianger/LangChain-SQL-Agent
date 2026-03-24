# LangChain SQL Agent 评估

基于 **SQLDatabaseToolkit + LangGraph (ReAct tool-calling)** 的 LangChain SQL Agent，
在 **LoCoMo** 和 **SyllabusQA** 数据集上进行评估。

## 项目结构

```
├── config.py           # 配置 (API, 路径, 参数)
├── token_tracker.py    # tiktoken (cl100k_base) token 计数 + LangChain 回调
├── metrics.py          # 评估指标: token-level F1, Recall, Accuracy
├── sql_agent.py        # SQL Agent 构建 (SQLDatabaseToolkit + LangGraph)
├── run_locomo.py       # LoCoMo 数据集评估流程
├── run_syllabusqa.py   # SyllabusQA 数据集评估流程
├── main.py             # 主入口
├── requirements.txt    # Python 依赖
└── results/            # 评估结果 (JSON)
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 小样本测试 (默认 20 个样本)
python main.py

# 3. 指定数据集和样本大小
python main.py --dataset locomo --sample-size 10

# 4. 全量测试
python main.py --full

# 5. 查看 Agent 中间步骤
python main.py --verbose
```

## 评估指标

| 指标 | 说明 |
|------|------|
| F1 Score | Token 级别 F1 (precision × recall 的调和平均) |
| Recall | Token 级别召回率 |
| Accuracy | 规范化后的精确匹配 |
| 检索时间 | Agent 从 SQL 数据库中查询并回答问题的平均时间 |
| 检索 Token 成本 | 使用 tiktoken cl100k_base 统计的 prompt + completion tokens |
| 插入时间/成本 | 数据准备阶段的插入耗时和 token 消耗（非只读 Agent 执行） |
| 删除时间/成本 | 数据准备阶段的删除耗时和 token 消耗（非只读 Agent 执行） |

## 架构说明

- **模型**: doubao-seed-1-8-251228 (Volcano Engine ARK API)
- **Token 计数**: tiktoken cl100k_base 编码器
- **Agent**: LangGraph 状态图 (assistant -> tools -> assistant) 的 ReAct 工具调用
- **数据库**: SQLite (通过 SQLAlchemy)
- **工具集**: SQLDatabaseToolkit (sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker) 四工具只读模式

## 数据集

### LoCoMo
- 10 个多轮对话样本, 共 5882 个对话轮次
- 1986 个 QA 问答对 (5 类: single-hop, temporal, multi-hop, open-domain, adversarial)
- 数据库表: `conversations`, `session_summaries`, `observations`

### SyllabusQA
- 1103 个测试 QA 对, 覆盖 13 个课程大纲
- 7 种问题类型: single/multi factual, single/multi reasoning, summarization, yes/no, no answer
- 数据库表: `syllabi` (全文), `syllabus_chunks` (分块)
