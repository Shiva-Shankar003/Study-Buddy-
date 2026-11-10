
import llm


def generate_answer(question, documents):
    """Generate an answer grounded in retrieved notes."""

    if not documents:
        return "Not found in the notes."

    context_parts = []

    for index, document in enumerate(documents, start=1):
        source = document.get("source", "Unknown source")
        content = document.get("text", "")

        context_parts.append(
            f"[{index}] Source: {source}\n{content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the user's question using only the supplied study notes.

Rules:
1. Do not add facts that are not supported by the notes.
2. Cite supporting notes using numbered citations such as [1] or [2].
3. If the notes do not contain the answer, say:
   Not found in the notes.
4. Give a clear and easy-to-understand answer.
5. Do not invent sources.

Study notes:
{context}

Question:
{question}

Answer:
"""

    answer = llm.ask(
        prompt,
        system="You are StudyBuddy, a study assistant that answers from notes."
    )

    return answer.strip()


if __name__ == "__main__":
    question = "What is normalization?"

    sample_documents = [
        {
            "source": "DBMS.txt",
            "text": "Normalization reduces data redundancy."
        }
    ]

    print(generate_answer(question, sample_documents))
