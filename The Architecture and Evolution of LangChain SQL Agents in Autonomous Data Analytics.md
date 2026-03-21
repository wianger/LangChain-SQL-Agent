# The Architecture and Evolution of LangChain SQL Agents in Autonomous Data Analytics

The paradigm of data interaction has undergone a fundamental transformation, shifting from deterministic, code-heavy interfaces to probabilistic, natural language-driven autonomous systems. At the forefront of this evolution is the LangChain SQL Agent, a sophisticated orchestration framework that enables large language models to interact directly with relational databases. This system functions not merely as a translator of text to Structured Query Language but as a comprehensive reasoning engine capable of schema exploration, query validation, iterative error recovery, and context-aware synthesis of complex datasets.[1] By bridging the gap between human conversational intent and the rigid structural requirements of SQL, these agents facilitate a new era of self-serve business intelligence, where the technical barriers to data access are systematically dismantled.[1]
## Architectural Foundations of Agentic SQL Interaction

To comprehend the nature of the LangChain SQL Agent, it is necessary to distinguish it from its predecessor, the SQL Chain. In the traditional chain architecture, the interaction was largely linear and sequential: a user provided a prompt, the system generated a query, and the result was returned.[2] This model, while efficient for rudimentary tasks, lacked the cognitive flexibility required to navigate real-world database environments characterized by ambiguous schemas and non-obvious table relationships. The SQL Agent, by contrast, adopts a reactive approach, often utilizing the ReAct (Reasoning and Acting) framework to plan and execute actions based on iterative observations of the database environment.[2, 3, 4]
The core of the agentic behavior lies in its ability to operate within a graph-based runtime, typically facilitated by LangGraph in contemporary implementations.[5, 6] This architecture decomposes the agent's task into a series of nodes—representing discrete steps such as model reasoning or tool execution—connected by edges that define the flow of information.[5, 7] This structure allows the agent to loop back to previous states when errors occur, ensuring a level of resilience and autonomy that static chains cannot achieve.[3, 8]

### Comparative Mechanics: Chains versus Agents

The transition from chains to agents represents a shift from a basic calculator to a scientific analytical partner.[2] While a SQL Chain might successfully retrieve the top five products from a single table if the schema is explicitly provided in the prompt, the SQL Agent possesses the intelligence to autonomously discover which tables are relevant, describe their schemas to understand foreign key relationships, and revise its strategy if the initial query fails.[1, 4, 9]
| **Feature Metric**   | **SQLDatabaseChain (Legacy)**        | LangChain SQL Agent (Modern)                   |
| -------------------- | ------------------------------------ | ---------------------------------------------- |
| **Logic Execution**  | Sequential / Linear                  | Iterative / Reactive (ReAct) [3, 8]            |
| **Schema Discovery** | Static (Manual Injection)            | Dynamic (Autonomous Exploration) [1, 2]        |
| **Error Handling**   | Fatal (Execution stops on error)     | Resilient (Self-correction loops) [3, 9]       |
| **Complexity Limit** | Low (Single-table/Simple aggregates) | High (Complex joins/Schema exploration) [1, 2] |
| **State Management** | Stateless per request                | Stateful (Graph-based persistence) [5, 10]     |
| **Efficiency**       | High for simple, known tasks         | High for ambiguous, multi-step tasks [2]       |

The superiority of the agentic approach is particularly evident in resolution rates. Research indicates that workflows built with multi-agent designs or sophisticated agentic loops see a 35%-45% increase in task resolution compared to single-agent or static-chain bots.[1] This improvement is attributed to the agent's ability to handle ambiguity and infer user intent from conversational language rather than requiring a rigid, command-line-style input.[1]
## The Core Components of the SQL Agentic Ecosystem

The functional capability of a LangChain SQL Agent is derived from the integration of three primary components: the language model reasoning engine, the database utility wrapper, and the specialized toolkit.[1, 9, 11]

### The Language Model as Reasoning Engine

The large language model serves as the "brain" or executor of the system.[1] It is responsible for parsing the user's natural language question, formulating a multi-step plan, and interpreting the tool outputs to construct a final response.[1] In the 2026 technological landscape, models such as GPT-5, Claude 3.7, and DeepSeek R1 are preferred for their high reasoning capabilities and specialized knowledge of various SQL dialects.[12, 13, 14] These models function within a graph where they execute nodes like the "model node" to decide on the next action and the "tools node" to interact with the database.[5]

### The SQLDatabase Utility Wrapper

Interaction with the data layer is facilitated by the `SQLDatabase` utility, which typically wraps around SQLAlchemy.[1, 15] This wrapper provides a unified interface for the agent to inspect table structures, column definitions, and relationships across various database systems including SQLite, PostgreSQL, and MySQL.[1, 10, 16] This utility is the "magic piece" that allows the agent to "peek" at the database schema, providing the necessary context for query generation without requiring the developer to manually define every table relationship in the prompt.[1]
### The SQLDatabaseToolkit

The agent's ability to take action is restricted to the tools provided in its toolkit.[5, 17] The SQLDatabaseToolkit is a standardized collection of functions that allow the agent to perform the discovery and execution steps essential for SQL generation.[3, 9, 11]
| **Tool Class**         | **Core Responsibility**                     | **Input Specification**                     |
| ---------------------- | ------------------------------------------- | ------------------------------------------- |
| `ListSQLDatabaseTool`  | Lists all table names within the database   | Typically an empty string [15, 18]          |
| `InfoSQLDatabaseTool`  | Retrieves metadata, schema, and sample rows | Comma-separated list of table names [9, 15] |
| `QuerySQLDatabaseTool` | Executes a detailed and correct SQL query   | A valid SQL SELECT query string [15, 19]    |
| `QuerySQLCheckerTool`  | Validates SQL syntax and logic using an LLM | SQL query string to be checked [15, 18]     |

The presence of the `QuerySQLCheckerTool` is particularly vital, as it introduces an internal validation layer where a secondary "SQL expert" model reviews the generated query for common mistakes like improper null handling or join columns before the query ever reaches the database engine.[3, 18, 20]
### Operational Workflow: The Iterative Reasoning Loop

The execution of a natural language query by a LangChain SQL Agent follows a rigorous, multi-step workflow designed to ensure accuracy and safety. This process is inherently iterative, allowing the agent to refine its understanding of the data as it progresses through the task.[3, 8]
#### Step 1: Schema Discovery and Discovery Phase

The agent typically begins by invoking the `sql_db_list_tables tool`.[9] This provides a list of all available tables, allowing the agent to understand the breadth of the database.[3, 8] In large enterprise environments, this initial list acts as the starting point for a "discovery" phase where the model matches the user's intent to the available data structures.[1, 16]

#### Step 2: Relevance Assessment and Detail Retrieval

Once the tables are listed, the agent decides which ones are most relevant to the question.[3, 8] It then calls the `sql_db_schema` tool (or `InfoSQLDatabaseTool`) for those specific tables.[3, 9] This tool returns the `CREATE TABLE` statements, providing the exact column names, data types, and—crucially—sample rows.[9, 18] The inclusion of sample rows is a fundamental best practice, as it helps the model understand the specific format of the data, such as whether a "status" column uses integers or string labels.[1, 9]
#### Step 3: Query Generation and Internal Validation

Using the schema information, the model generates a SQL query. It is typically prompted to use explicit column lists rather than `SELECT *` and to apply a result limit (e.g., top_k=5) unless instructed otherwise.[3, 8, 21] Before execution, the agent uses the `sql_db_query_checker` tool to analyze the query for logic errors, such as joining on columns with mismatched data types or failing to account for case sensitivity in predicates.[3, 18]
#### Step 4: Execution and Self-Correction Loop

The validated query is executed against the database.[3, 8] If the database returns an error—such as a `sqlite3.OperationalError` due to a non-existent column—the agent does not fail.[3, 9] Instead, it receives the error message, reasons about why the query failed, and uses its tools to gather missing information.[4, 9] For example, if it hallucinated a column name, it may call sql_db_schema again to verify the correct nomenclature before rewriting and re-executing the query.[3, 9] This feedback loop of providing the model with error messages is a powerful pattern for building resilient systems.[3, 8]
## Advanced Context Management: Progressive Disclosure

As databases scale to include thousands of tables and complex vertical-specific business logic, it becomes impossible to include the entire schema in a single prompt.[22] Stuffing the context window with irrelevant table definitions not only increases costs but also introduces "distractions" that lead to poorer tool selection and increased reasoning latency.[23, 24] To mitigate this, modern SQL agents employ the "Progressive Disclosure" pattern.[22]
### The Three-Layer Context Architecture

Progressive disclosure organizes database capabilities into hierarchical levels of detail, loading information only when the agent determines it is necessary for the current task.[22, 24]

1. Metadata Layer (Discovery): The system prompt contains only lightweight descriptions of available "skills" or table clusters (e.g., "Sales Analytics," "Inventory Management").[22, 24]
2. Activation Layer (Schema Loading): When the agent identifies a relevant skill based on the user's query, it invokes a tool like load_skill to fetch the specific schemas and business logic required for that domain.[22, 24]
3. Execution Layer (Detailed Reference): Only when the agent is ready to write the query does it pull in additional assets like few-shot examples or complex join hints specific to those tables.[24, 25]
   

This modular architecture allows enterprises to scale their AI assistants to hundreds of independent business units without overwhelming the model's working memory.[22] By engaging compression steps at threshold fractions of the context window size—often around 85%—the system ensures that high-signal tokens are prioritized while older, redundant tool outputs are offloaded or summarized.[23, 26]
### Performance and Reliability Metrics

Performance benchmarking in 2026 suggests that while SQL agents are highly capable, their latency is significantly affected by the complexity of the reasoning required.

| **Performance Metric**    | **Expected Value / Trend** | **Impact Factor**                              |
| ------------------------- | -------------------------- | ---------------------------------------------- |
| **Simple Query Latency**  | 5-15 seconds               | Direct schema retrieval [27]                   |
| **Complex Query Latency** | 2-3 minutes                | Multiple rewrite/correction loops [27]         |
| **Few-Shot Accuracy**     | 16%→52% (Claude 3)         | Using 3 semantically similar examples [28]     |
| **Resolution Rate Delta** | +35% to +45%               | Multi-agent vs. single-agent design [1]        |
| **Context Threshold**     | 20,000 tokens              | Trigger for offloading large tool results [23] |

To optimize these times, developers are encouraged to use "data dictionaries"—static metadata layers that describe table attributes and meanings in natural language—which can reduce complex resolution times from minutes to seconds.[27]

## Security Governance and the Zero-Trust Framework

The execution of model-generated SQL presents profound security challenges. A clever prompt injection could potentially trick an agent into deleting records or exfiltrating sensitive PII.[29, 30] Therefore, a production-ready SQL agent must be designed under a zero-trust model, treating the agent as an untrusted user rather than a trusted component of the system.[31]

### Core Security Principles for SQL Agents

The foundation of secure agent design is the rigorous application of role-based access control (RBAC) and least-privilege credentials.[31, 32]

- Scoped Permissions: The database user associated with the agent should be restricted to a read-only replica with access only to the necessary tables.[8, 31, 33]
- DML and DDL Prevention: The agent must be explicitly forbidden from executing any command other than SELECT. This is enforced through both system prompting and database-level permissions that block INSERT, UPDATE, DELETE, DROP, and ALTER.[3, 8, 21]
- Prompt Injection Detection: Frameworks like Rebuff are integrated to provide an "LLM guard," analyzing incoming prompts for malicious patterns before they reach the reasoning engine.[29, 30] Rebuff utilizes heuristics, dedicated detection models, and "canary tokens" to identify and block attacks.[30]
- Parameterization: Whenever possible, queries should be parameterized to prevent traditional SQL injection vulnerabilities that could arise from the LLM inadvertently embedding raw user input into a query string.[20, 32]

### Human-in-the-Loop and Approval Workflows

For mission-critical applications, LangChain facilitates a "Human-in-the-Loop" architecture.[3, 7] In this pattern, the agent reaches a state where it has generated a query but pauses execution to wait for human approval.[3, 12] Using LangGraph checkpointers, the system saves its current state (including the conversation history and the proposed SQL), allowing a human analyst to review, edit, or reject the query through a UI before it is executed.[3, 7, 12] This provides a final safeguard against hallucinations or unintended actions in high-stakes environments like healthcare or finance.[19, 34]
## Implementation Strategies: LangChain versus LangGraph

Developers building SQL agents must choose between high-level abstractions and low-level workflow control. While create_sql_agent provides a production-ready implementation with minimal code, it relies heavily on the system prompt to constrain behavior.[3, 7, 8]
### Customization via LangGraph

Implementing a SQL agent directly in LangGraph allows for dedicated nodes that force specific tool calls.[7] This is particularly useful when developers want to ensure the agent follows a strict protocol, such as "Always check the table list before querying any schema".[3, 8]

1. Node: List Tables: Initially identifies the available data landscape.
2. Node: Get Schema: Forces the agent to retrieve specific table structures.[7]
3. Node: Generate Query: The core reasoning step where the LLM writes the SQL.
4. Node: Check Query: A separate node where an LLM expert validates the query for common mistakes.[7]
5. Node: Execution: The final step where the query is run against the database.
   

By putting these steps into dedicated nodes, developers can customize the prompts for each individual phase and implement fine-grained routing logic, such as a conditional edge that returns the agent to the "Generate Query" node if the "Check Query" node finds an error.[7]
### Prompt Engineering and Few-Shot Learning

The real power of a LangChain SQL agent lies in its ability to understand what the user is trying to achieve.[1] The quality of the generated SQL is a direct reflection of the prompt's quality.[1] Effective prompts for SQL agents often include:

- Chain-of-Thought (CoT): Encouraging the agent to "think step-by-step" before writing the final query. This helps it break down the problem into smaller logical pieces like identifying tables, figuring out joins, and then applying filters.[1]
- Template Modification: Permanently adding business rules or specific context (e.g., "For productive time, use minutes as units") to the underlying prompt template.[1, 21]
- Few-Shot Examples: Providing a list of (Input, SQL) pairs between the system prompt and the user question.[1, 28] Research indicates that formatting these examples as messages rather than a single long string produces significantly better results for models like Claude 3.[28]
  
## Real-World Applications and Industry Use Cases

The LangChain SQL Agent has found significant traction across sectors that rely on real-time data analysis and complex relational structures.
### Manufacturing and Industrial Intelligence

In the manufacturing sector, SQL agents are utilized to monitor production efficiency and bottleneck identification.[21] Specialized system prompts define "Manufacturing Expert" personas that understand how to map "productive time" and "downtime" to specific tables like status and product_tracking.[21] These agents can handle complex questions regarding average processing times or identifying which machines are currently causing delays without requiring the floor manager to understand the underlying schema.[21]
### Healthcare and Clinical Knowledge Management

Healthcare organizations leverage SQL agents to navigate clinical guidelines and anonymized patient records while maintaining strict privacy standards.[34] Clinical assistants can help physicians find relevant treatment protocols or identify trends in anonymized patient outcomes by querying medical knowledge bases.[34] The ability of the agent to maintain context across multi-turn conversations is particularly useful for clinical research tasks, where initial questions are often followed by requests for more granular details.[34]
### Business Intelligence and E-commerce

Retailers use SQL agents to provide style and compatibility advice by querying product catalogs and unified customer profiles.[34] A marketing team can use the agent to pull campaign performance numbers on the fly during a strategy session, allowing them to adjust their tactics based on real-time revenue data.[1] This democratization of data access ensures that insights are available to the decision-makers who need them most, reducing the reliance on central data engineering teams for routine reporting.[1]
## The 2026 Ecosystem: Interoperability and Deep Agents

The landscape of SQL interaction is currently being reshaped by the Model Context Protocol (MCP) and the emergence of "Deep Agents".[16, 22]
### The Role of MCP Servers

The Model Context Protocol (MCP) is an open standard that enables AI applications to securely access local tools and databases.[16] By building an MCP server, developers can expose database schema and query tools to any AI client, such as Claude Desktop, facilitating seamless local database interactions.[16] This protocol allows for a "plugin-like" experience where an agent can discover and utilize a database as a standardized skill, further promoting the modular architecture of autonomous systems.[16]
### Deep Agents and Context Optimization

Deep Agents represent a further evolution, focusing on advanced context engineering and offloading strategies.[23, 26] These agents are designed for high-impact use cases where workflows have branching logic and partial failure is expected.[35] They employ sophisticated memory management, classifying agent memory into short-term (working context) and long-term persistent state (facts, prior decisions, and tool outputs stored in databases).[26] As the session context approaches limits, these agents truncate older tool calls and replace them with pointers to files on disk, ensuring that the model always receives high-signal tokens within its fixed token budget.[23, 26]
## Technical Evaluation and Deployment Patterns

Building a reliable SQL agent requires a robust evaluation framework. LangSmith is the standard tool for tracing, debugging, and evaluating agent behavior.[1, 5, 9] Evaluation typically focuses on three distinct metrics:

- Final Response Evaluation: Assessing whether the agent's natural language answer correctly addresses the user's prompt.[36]
- Trajectory Evaluation: Comparing the actual sequence of tool calls against an expected "golden" path. For example, verifying if the agent called sql_db_list_tables before sql_db_query.[36]
- Single-Step Evaluation: Isolating an individual agent step to determine if it selected the appropriate tool for a specific context.[36]

### Observability and Error Management

Logging the full action chain—including the exact prompt received, the model's internal "thinking," and any self-correction attempts—is critical for production systems.[1, 31] This visibility allows developers to spot weird behavior patterns, such as an agent getting stuck in a loop making repetitive API calls, and to implement individual "circuit breakers" that throttle tools automatically.[31]

| **Evaluation Layer**    | **Key Focus Area**                     | **Tools Utilized**                  |
| ----------------------- | -------------------------------------- | ----------------------------------- |
| **Logic Layer**         | Syntax correctness and join logic      | `QuerySQLCheckerTool` [18]          |
| **Execution Layer**     | Query performance and result accuracy  | `QuerySQLDatabaseTool` [15]         |
| **Safety Layer**        | DML prevention and injection detection | Rebuff / RBAC [30, 31]              |
| **Observability Layer** | Step-by-step tracing and trajectory    | LangSmith [5, 36]                   |
| **User Experience**     | Final response clarity and memory      | `ConversationBufferMemory` [10, 32] |

## **Conclusions and Strategic Recommendations**

The LangChain SQL Agent is a fundamental component of the modern AI tech stack, enabling a transformative shift in how organizations access and utilize their data. By wrapping relational databases in an agentic reasoning framework, it allows for a more intuitive, conversational, and autonomous approach to data analysis. However, the power of these systems must be balanced with rigorous security governance and sophisticated context management.

Strategic recommendations for successful deployment include the implementation of progressive disclosure for large-scale schemas to maintain model focus and minimize costs. Furthermore, developers should prioritize the LangGraph implementation for production environments to ensure deterministic control over discovery and validation steps. Security must be managed through a zero-trust architecture, combining restricted database permissions with active prompt injection detection. Finally, the use of evaluation tools like LangSmith is essential to move from "vibes-based" development to production-grade reliability, ensuring that the agent remains a trusted and effective partner in the analytical process. As the ecosystem moves toward deeper interoperability via MCP and increasingly capable reasoning models, the LangChain SQL Agent will remain the definitive bridge between the language of humans and the data of machines.

Master the LangChain SQL Agent in Your Next Project - Explore data at any technical level, https://querio.ai/blogs/langchain-sql-agent
Difference Between SQLChain and SQLAgent | by Ashish Malhotra - Medium, https://medium.com/@mrcoffeeai/difference-between-sqlchain-and-sqlagent-0804397bb30c
Build a SQL agent - Docs by LangChain, https://docs.langchain.com/oss/python/langchain/sql-agent
What is the difference between 'SQLDatabaseChain' and 'create_sql_agent' in langchain?, https://stackoverflow.com/questions/76920224/what-is-the-difference-between-sqldatabasechain-and-create-sql-agent-in-lang
Agents - Docs by LangChain, https://docs.langchain.com/oss/javascript/langchain/agents
Agents - Docs by LangChain, https://docs.langchain.com/oss/python/langchain/agents
Build a custom SQL agent - Docs by LangChain, https://docs.langchain.com/oss/python/langgraph/sql-agent
Build a SQL agent - Docs by LangChain, https://docs.langchain.com/oss/javascript/langchain/sql-agent
SQLDatabase toolkit integration - Docs by LangChain, https://docs.langchain.com/oss/python/integrations/tools/sql_database
Building a Conversational SQL Agent with LangChain and FastAPI - Medium, https://medium.com/@silverskytechnology/building-a-conversational-sql-agent-with-langchain-and-fastapi-7fb2c96228a5
SQLDatabaseToolkit | langchain_community - LangChain Reference Docs, https://reference.langchain.com/python/langchain-community/agent_toolkits/sql/toolkit/SQLDatabaseToolkit
LangChain Python Tutorial: A Complete Guide for 2026 | The PyCharm Blog, https://blog.jetbrains.com/pycharm/2026/02/langchain-tutorial-2026/
We benchmarked 19 popular LLMs on SQL generation with a 200M row dataset - Reddit, https://www.reddit.com/r/dataengineering/comments/1khsiwd/we_benchmarked_19_popular_llms_on_sql_generation/
ai-agent-demo/agent_demo_deepseek.ipynb at main - GitHub, https://github.com/backblaze-b2-samples/ai-agent-demo/blob/main/agent_demo_deepseek.ipynb
toolkit | langchain_community - LangChain Reference Docs, https://reference.langchain.com/python/langchain-community/agent_toolkits/sql/toolkit
Building AI Agents That Query SQL Databases — Two Practical Methods (MCP Server & LangChain) | by Ossama El Sanharawi | Medium, https://medium.com/@elsossama/building-ai-agents-that-query-sql-databases-two-practical-methods-mcp-server-langchain-00d5007d6e05
Unlocking the Power of LangChain's SQL Agent: A Deep Dive into Natural Language Database Interactions | by Syed Muhammed Hassan Ali | Medium, https://medium.com/@syed007hassan/unlocking-the-power-of-langchains-sql-agent-a-deep-dive-into-natural-language-database-4b2b2dcd6d18
LangChain: SQLDatabase Built-In Toolkit Guide - Kaggle, https://www.kaggle.com/code/ksmooi/langchain-sqldatabase-built-in-toolkit-guide
Empowering Data-Driven Decisions: Embedding Trust in Text-to-SQL AI Agents, https://towardsdatascience.com/embedding-trust-into-text-to-sql-ai-agents-3f15d0ddaf1a/
Text2SQL Best Practices. Natural Language Data Queries | by Xin Cheng - Medium, https://billtcheng2013.medium.com/text2sql-best-practices-d25e5ed19b24
SQL Agent with Cohere and LangChain (i-5O Case Study), https://docs.cohere.com/page/sql-agent-cohere-langchain
Build a SQL assistant with on-demand skills - Docs by LangChain, https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant
Context Management for Deep Agents - LangChain Blog, https://blog.langchain.com/context-management-for-deepagents/
Progressive Disclosure: the technique that helps control context (and tokens) in AI agents, https://medium.com/@martia_es/progressive-disclosure-the-technique-that-helps-control-context-and-tokens-in-ai-agents-8d6108b09289
Skills - Docs by LangChain, https://docs.langchain.com/oss/python/langchain/multi-agent/skills
Comparing File Systems and Databases for Effective AI Agent Memory Management, https://blogs.oracle.com/developers/comparing-file-systems-and-databases-for-effective-ai-agent-memory-management
Sql Agent performance : r/LangChain - Reddit, https://www.reddit.com/r/LangChain/comments/1g10t65/sql_agent_performance/
Few-shot prompting to improve tool-calling performance - LangChain Blog, https://blog.langchain.com/few-shot-prompting-to-improve-tool-calling-performance/
Enhancing security in text-to-SQL systems: A novel dataset and agent-based framework | Natural Language Processing | Cambridge Core, https://www.cambridge.org/core/journals/natural-language-processing/article/enhancing-security-in-texttosql-systems-a-novel-dataset-and-agentbased-framework/4D0F32A20438C18FD1F84DC7BD908F97
Rebuff: Detecting Prompt Injection Attacks - LangChain Blog, https://blog.langchain.com/rebuff/
What I wish I knew about agent security before deploying to prod : r/LangChain - Reddit, https://www.reddit.com/r/LangChain/comments/1pbknpj/what_i_wish_i_knew_about_agent_security_before/
How do I handle data privacy and security when using LangChain? - Milvus, https://milvus.io/ai-quick-reference/how-do-i-handle-data-privacy-and-security-when-using-langchain
[D] How to prevent SQL injection in LLM based Text to SQL project ? : r/MachineLearning, https://www.reddit.com/r/MachineLearning/comments/1ff1y95/d_how_to_prevent_sql_injection_in_llm_based_text/
Langchain in the Real World: Case Studies and Success Stories - Mukesh Yadav, https://www.mukeshyadav.com/blog/langchain-real-world-applications
LangChain Use Cases: Practical Patterns for Building Reliable LLM Agents, https://techtidesolutions.com/blog/langchain-use-cases/
Evaluate a complex agent - Docs by LangChain, https://docs.langchain.com/langsmith/evaluate-complex-agent