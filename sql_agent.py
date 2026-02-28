"""LangChain SQL Agent: SQLDatabaseToolkit + AgentExecutor (ReAct tool-calling).

Uses ``create_tool_calling_agent`` with ``AgentExecutor`` so the model
invokes SQL tools via structured function calls rather than fragile
text-based ReAct parsing — much more reliable with doubao / OpenAI-compat
APIs while keeping the same toolkit + executor architecture.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, Tool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, event, text

import config
from token_tracker import TokenTracker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise SQL database assistant. "
    "Use the provided tools to explore the database schema and run queries. "
    "ALWAYS use the tools to get real data — never fabricate results. "
    "If the answer cannot be determined from the database, say 'No/insufficient information'."
)


def get_llm(callbacks: Optional[List] = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        temperature=0,
        callbacks=callbacks,
    )


def _sqlite_engine(db_path: str):
    """Create a SQLite engine with WAL mode and busy timeout for concurrency."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=10000")
        cur.close()

    return engine


def _make_write_tool(engine) -> Tool:
    """Tool that executes INSERT / UPDATE / DELETE statements."""

    def _run_write_sql(query: str) -> str:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                conn.commit()
                upper = query.strip().upper()
                if upper.startswith("SELECT"):
                    rows = result.fetchall()
                    return str(rows[:50])
                return f"OK – rows affected: {result.rowcount}"
        except Exception as exc:
            return f"Error: {exc}"

    return Tool(
        name="sql_db_write",
        description=(
            "Execute a SQL statement that MODIFIES the database "
            "(INSERT, UPDATE, DELETE). Returns confirmation or error. "
            "For read-only SELECT queries, prefer sql_db_query."
        ),
        func=_run_write_sql,
    )


def build_agent(
    db_path: str,
    token_tracker: TokenTracker,
    *,
    extra_tools: Optional[List[BaseTool]] = None,
    max_iterations: int = config.AGENT_MAX_ITERATIONS,
    verbose: bool = False,
) -> AgentExecutor:
    """Return an AgentExecutor wired to the given SQLite database."""
    engine = _sqlite_engine(db_path)
    db = SQLDatabase(engine=engine)
    llm = get_llm(callbacks=[token_tracker])

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools: List[BaseTool] = toolkit.get_tools()
    tools.append(_make_write_tool(engine))
    if extra_tools:
        tools.extend(extra_tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


def run_agent(executor: AgentExecutor, question: str) -> str:
    """Invoke the agent synchronously and return the final answer string."""
    try:
        result = executor.invoke({"input": question})
        return result.get("output", "")
    except Exception as exc:
        logger.warning("Agent error: %s", exc)
        return f"[ERROR] {exc}"


async def arun_agent(executor: AgentExecutor, question: str) -> str:
    """Invoke the agent in a thread (async-friendly) and return the answer."""
    try:
        result = await asyncio.to_thread(executor.invoke, {"input": question})
        return result.get("output", "")
    except Exception as exc:
        logger.warning("Agent error: %s", exc)
        return f"[ERROR] {exc}"
