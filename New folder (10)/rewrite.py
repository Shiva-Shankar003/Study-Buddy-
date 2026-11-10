
import llm


def rewrite_query(question):
    """Rewrite a question into a search-friendly query."""

    prompt = f"""
Rewrite the following question into one concise search query
for finding relevant study notes.

Rules:
- Preserve the original meaning.
- Keep important technical keywords.
- Do not answer the question.
- Return only the rewritten query.

Question: {question}
"""

    try:
        result = llm.ask(
            prompt,
            system="You are a search-query rewriting assistant."
        )

        return result.strip().strip('"')

    except Exception as error:
        print(f"Query rewriting failed: {error}")
        return question


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "Normalization kya hota hai?"

    print("Original query:", question)
    print("Rewritten query:", rewrite_query(question))
