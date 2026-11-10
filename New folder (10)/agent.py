"""
StudyBuddy - Agentic Question Answering Workflow

Flow: route -> rewrite -> retrieve -> grade -> (retry) -> generate -> verify
If the notes cannot answer the question, the agent answers from the AI's own
general knowledge and clearly says so (instead of just "Not found").
"""

import re
import sys
import time

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from llm import ask
from rewrite import rewrite_query
from retrieval import retrieve
from generate import generate_answer
from verify import verify_answer

NOT_FOUND = "Not found in the notes."
GENERAL_NOTE = ("*Note: this was not found in your notes, so this answer comes from "
                "general AI knowledge (not verified against your notes).*\n\n")

DEFAULTS = {"use_rewrite": True, "use_crag": True, "use_verify": True}

def _is_not_found(text):
    return (not text) or text.strip().lower().startswith("not found")

class AgentState(TypedDict, total=False):
    question: str
    subject: Optional[str]
    intent: str
    query: str
    chunks: list
    relevant: bool
    retries: int
    answer: str
    general: bool
    verification: dict
    error: str
    use_rewrite: bool
    use_crag: bool
    use_verify: bool

def route_question(state: AgentState):
    question = state["question"].lower()

    if re.search(r"\b(quiz|mcq|mcqs|test me|flashcards)\b", question):
        intent = "quiz"
    elif re.search(r"\b(summarize|summarise|summary|overview)\b", question):
        intent = "summary"
    else:
        intent = "qa"

    return {"intent": intent}

def rewrite_question(state: AgentState):
    question = state["question"]

    if not state.get("use_rewrite", True):
        return {"query": question}

    try:
        query = rewrite_query(question)
        if not query or not query.strip():
            query = question
    except Exception as error:
        print("Rewrite warning:", error)
        query = question

    return {"query": query}

def retrieve_notes(state: AgentState):
    query = state.get("query", state["question"])

    try:
        chunks = retrieve(query, top_k=8, subject=state.get("subject"))
    except Exception as error:
        print("Retrieval error:", error)
        chunks = []

    return {"chunks": chunks, "relevant": bool(chunks)}

def grade_relevance(state: AgentState):
    chunks = state.get("chunks", [])

    if not chunks:
        return {"relevant": False}

    if not state.get("use_crag", True):
        return {"relevant": True}

    context = "\n\n".join(chunk.get("text", "")[:1200] for chunk in chunks[:5])

    prompt = f"""
Decide whether the following study notes contain useful
information to answer the student's question.

Reply with exactly YES or NO.

Question:
{state['question']}

Study notes:
{context}
"""

    try:
        verdict = ask(prompt, system="You are a strict study-notes relevance judge.")
        relevant = verdict.strip().upper().startswith("YES")
    except Exception as error:
        print("Relevance check warning:", error)
        relevant = bool(chunks)

    return {"relevant": relevant}

def decide_after_grade(state: AgentState):
    retries = state.get("retries", 0)

    if state.get("relevant", False):
        return "generate"

    if retries < 1:
        return "retry"

    return "generate"

def retry_search(state: AgentState):
    retries = state.get("retries", 0) + 1
    query = state["question"] + " definition meaning explanation"
    return {"query": query, "retries": retries}

def general_answer(question):
    """Answer from the AI's own knowledge when the notes do not help."""
    try:
        text = ask(
            f"A student asked: {question}\n\n"
            "Answer clearly in simple language. Give the standard definition, "
            "the key points and a short example if useful.",
            system="You are StudyBuddy, a friendly and accurate study tutor.",
        )
    except Exception as error:
        print("General answer warning:", error)
        return NOT_FOUND

    text = (text or "").strip()
    return GENERAL_NOTE + text if text else NOT_FOUND

def create_answer(state: AgentState):
    chunks = state.get("chunks", [])
    answer = NOT_FOUND

    if chunks and state.get("relevant", False):
        try:
            answer = generate_answer(state["question"], chunks)
        except Exception as error:
            print("Answer generation error:", error)
            answer = NOT_FOUND

    if _is_not_found(answer):
        return {"answer": general_answer(state["question"]), "general": True}

    return {"answer": answer, "general": False}

def check_answer(state: AgentState):
    answer = state.get("answer", NOT_FOUND)
    chunks = state.get("chunks", [])

    skipped = {
        "kept_answer": answer,
        "claims": [],
        "faithfulness": None,
        "citation_precision": None,
        "verified": False,
    }

    if state.get("general", False):
        skipped["general"] = True
        return {"verification": skipped}

    if _is_not_found(answer) or not chunks or not state.get("use_verify", True):
        return {"verification": skipped}

    try:
        result = verify_answer(answer, chunks)
    except Exception as error:
        print("Verification warning:", error)
        result = skipped

    return {"verification": result}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_question)
    graph.add_node("rewrite", rewrite_question)
    graph.add_node("retrieve", retrieve_notes)
    graph.add_node("grade", grade_relevance)
    graph.add_node("retry", retry_search)
    graph.add_node("generate", create_answer)
    graph.add_node("verify", check_answer)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        lambda state: "stop" if state["intent"] == "quiz" else "continue",
        {"stop": END, "continue": "rewrite"},
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        decide_after_grade,
        {"retry": "retry", "generate": "generate"},
    )

    graph.add_edge("retry", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()

_GRAPH = None

def run(question, subject=None, **overrides):
    """
    Returns: intent, answer, chunks, verification, latency.
    Optional switches (to make it faster): use_rewrite, use_crag, use_verify
    (all default True). Other keyword arguments are accepted and ignored.
    """

    global _GRAPH

    if _GRAPH is None:
        _GRAPH = build_graph()

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    start = time.time()

    initial_state = {
        "question": question,
        "subject": subject,
        "retries": 0,
        "use_rewrite": overrides.get("use_rewrite", DEFAULTS["use_rewrite"]),
        "use_crag": overrides.get("use_crag", DEFAULTS["use_crag"]),
        "use_verify": overrides.get("use_verify", DEFAULTS["use_verify"]),
    }

    result = _GRAPH.invoke(initial_state)

    if result.get("intent") == "quiz":
        answer = "Quiz generation will be handled by the quiz module."
        chunks = []
        verification = {}
    else:
        verification = result.get("verification", {})
        answer = verification.get("kept_answer", result.get("answer", NOT_FOUND))
        chunks = [] if verification.get("general") else result.get("chunks", [])

    return {
        "intent": result.get("intent", "qa"),
        "answer": answer,
        "chunks": chunks,
        "verification": verification,
        "latency": time.time() - start,
    }

def plain_answer(question):
    start = time.time()

    answer = ask(question, system="Answer the student's question concisely.")

    return {
        "intent": "qa",
        "answer": answer,
        "chunks": [],
        "verification": {},
        "latency": time.time() - start,
    }

if __name__ == "__main__":

    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "What is normalization?"

    try:
        result = run(question)

        print("\nINTENT:", result["intent"])

        if result["intent"] == "quiz":
            print("\n" + result["answer"])
        else:
            print("\nANSWER:\n")
            print(result["answer"])

            print("\nSOURCES:\n")

            for index, chunk in enumerate(result.get("chunks", []), start=1):
                source = chunk.get("source", chunk.get("ref", "Unknown source"))
                print(f"[{index}] {source}")

            verification = result.get("verification", {})
            faithfulness = verification.get("faithfulness")
            precision = verification.get("citation_precision")

            if faithfulness is not None:
                print(f"\nFaithfulness: {faithfulness:.2f}")

            if precision is not None:
                print(f"Citation precision: {precision:.2f}")

        print(f"\nLatency: {result['latency']:.2f} seconds")

    except Exception as error:
        print("\nStudyBuddy error:", error)
        raise
