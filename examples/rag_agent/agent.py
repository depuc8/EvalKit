"""A small LangGraph RAG agent: retrieve -> (maybe) calculate -> generate.

Run the evaluation with::

    evalkit run --dataset examples/rag_agent/golden.jsonl \\
        --agent examples.rag_agent.agent:agent --pricing examples/rag_agent/pricing.json
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from evalkit.adapters.langgraph import LangGraphAdapter, replayable
from examples.rag_agent.fake_model import ExtractiveChatModel
from examples.rag_agent.knowledge_base import BM25, DOCS

SYSTEM_PROMPT = (
    "You are a support assistant for Acme Cloud. Answer using only the provided context. "
    "If the context does not contain the answer, say you do not know."
)

_INDEX = BM25(DOCS)


class KBRetriever(BaseRetriever):
    """LangChain retriever over the toy knowledge base."""

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
        from langchain_core.documents import Document

        return [Document(page_content=text) for text in _INDEX.search(query)]


@tool
def search_kb(query: str) -> str:
    """Search the Acme Cloud knowledge base; returns the best passages separated by blank lines."""
    return "\n\n".join(d.page_content for d in KBRetriever().invoke(query))


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> float:
    """Evaluate a basic arithmetic expression such as '49*5*12'."""
    return _eval(ast.parse(expression, mode="eval").body)


class State(TypedDict, total=False):
    question: str
    context: list[str]
    calculation: str
    answer: str


def retrieve(state: State) -> State:
    found = search_kb.invoke({"query": state["question"]})
    return {"context": [p for p in found.split("\n\n") if p]}


def _money(value: float) -> str:
    return f"${value:,.0f}" if float(value).is_integer() else f"${value:,.2f}"


def compute(state: State) -> State:
    """Answer cost questions with the calculator tool, using prices found in the context."""
    q, context = state["question"], " ".join(state.get("context", []))
    expression = label = None
    seats = re.search(r"(\d+)\s+(?:pro\s+)?(?:seats?|users?)", q, re.I)
    storage = re.search(r"(\d[\d,]*)\s*GB", q, re.I)
    if seats and (price := re.search(r"\$(\d+(?:\.\d+)?) per user per month", context)):
        n, yearly = int(seats.group(1)), bool(re.search(r"year|annual", q, re.I))
        expression = f"{price.group(1)}*{n}" + ("*12" if yearly else "")
        label = f"{n} Pro {'seats' if 'seat' in q.lower() else 'users'} cost {{}} per " + (
            "year" if yearly else "month"
        )
    elif (
        storage
        and "extra storage" in q.lower()
        and (price := re.search(r"\$(\d+(?:\.\d+)?) per GB per month", context))
    ):
        n = int(storage.group(1).replace(",", ""))
        expression = f"{price.group(1)}*{n}"
        label = f"{n} GB of extra storage on Pro costs {{}} per month"
    if expression is None or label is None:
        return {}
    value = calculator.invoke({"expression": expression})
    return {"calculation": label.format(_money(value)) + "."}


_llm = replayable(ExtractiveChatModel(), factory=lambda name: ExtractiveChatModel(model_name=name))


def generate(state: State) -> State:
    passages = "\n".join(f"[{i}] {p}" for i, p in enumerate(state.get("context", []), 1))
    prompt = f"Context:\n{passages}\n\nQuestion: {state['question']}"
    if state.get("calculation"):
        prompt += f"\nCalculation: {state['calculation']}"
    reply = _llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
    return {"answer": str(reply.content)}


def build_graph() -> Any:
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("compute", compute)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "compute")
    builder.add_edge("compute", "generate")
    builder.add_edge("generate", END)
    return builder.compile(name="acme_rag")


graph = build_graph()

#: The graph wrapped for EvalKit; ``calculator`` results count as retrieved context so that
#: computed numbers are not flagged as unsupported by the groundedness scorer.
agent = LangGraphAdapter(graph, context_tools={"calculator"})
