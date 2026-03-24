"""LangChain SQL Agent: SQLDatabaseToolkit + LangGraph ReAct graph.

Builds a small LangGraph loop (assistant -> tools -> assistant) where the
model uses structured tool calls to interact with SQLDatabaseToolkit tools.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
from sqlalchemy import create_engine, event

import config
from token_tracker import TokenTracker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise SQL database assistant. "
    "Use the provided tools to explore the database schema and run queries. "
    "ALWAYS use the tools to get real data — never fabricate results. "
    "If the answer cannot be determined from the database, say 'No/insufficient information'."
)


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    iterations: int


def get_llm(callbacks: Optional[List] = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.MODEL_NAME,
        api_key=SecretStr(config.API_KEY),
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


def build_agent(
    db_path: str,
    token_tracker: TokenTracker,
    *,
    extra_tools: Optional[List[BaseTool]] = None,
    max_iterations: int = config.AGENT_MAX_ITERATIONS,
    verbose: bool = False,
) -> Any:
    """Return a compiled LangGraph SQL agent wired to the given SQLite database."""
    engine = _sqlite_engine(db_path)
    db = SQLDatabase(engine=engine)
    llm = get_llm(callbacks=[token_tracker])

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools: List[BaseTool] = toolkit.get_tools()
    if extra_tools:
        logger.warning(
            "Strict read-only mode enabled: ignoring extra_tools to keep four-tool action space"
        )

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def _assistant(state: AgentState) -> Dict[str, Any]:
        msgs = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(msgs)
        if verbose:
            logger.info("Assistant response: %s", response)
        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }

    def _route_after_assistant(state: AgentState) -> str:
        if state.get("iterations", 0) >= max_iterations:
            if verbose:
                logger.warning(
                    "Reached max iterations (%d), stopping graph", max_iterations
                )
            return "end"

        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("assistant", _assistant)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        _route_after_assistant,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "assistant")
    return graph.compile()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(str(item.get("text", item)))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    return str(content)


def _extract_retrieved_texts(result: Dict[str, Any]) -> List[str]:
    """Extract sql_db_query tool observations from LangGraph message history."""
    texts: List[str] = []
    for msg in result.get("messages", []):
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = getattr(msg, "name", None) or msg.additional_kwargs.get("name")
        if tool_name != "sql_db_query":
            continue
        observation = _content_to_text(msg.content)
        obs_stripped = observation.strip()
        if obs_stripped and not obs_stripped.startswith(("OK", "Error")):
            texts.append(obs_stripped)
    return texts


def _extract_output(result: Dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return _content_to_text(msg.content)
    return ""


def run_agent(graph: Any, question: str) -> tuple[str, List[str]]:
    """Invoke the LangGraph agent and return (answer, retrieved_texts)."""
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)], "iterations": 0}
        )
        return _extract_output(result), _extract_retrieved_texts(result)
    except Exception as exc:
        logger.warning("Agent error: %s", exc)
        return f"[ERROR] {exc}", []


async def arun_agent(graph: Any, question: str) -> tuple[str, List[str]]:
    """Invoke the LangGraph agent asynchronously and return (answer, retrieved_texts)."""
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)], "iterations": 0}
        )
        return _extract_output(result), _extract_retrieved_texts(result)
    except Exception as exc:
        logger.warning("Agent error: %s", exc)
        return f"[ERROR] {exc}", []
