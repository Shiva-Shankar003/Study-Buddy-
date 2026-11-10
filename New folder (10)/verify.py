
"""
StudyBuddy - Answer Verification Module

Responsibilities:
1. Split the generated answer into individual claims.
2. Identify source citations such as [1] and [2].
3. Ask the LLM to verify claims against the evidence.
4. Remove unsupported claims.
5. Calculate faithfulness and citation precision.
"""

import re

from llm import ask_json


# Standard response when the notes do not contain the answer.
NOT_FOUND = "Not found in the notes."

# Detect citations such as [1], [2], [3].
CITE = re.compile(r"\[(\d+)\]")


def split_claims(answer):
    """
    Split an answer into individual sentences or lines.

    Example:
        "DBMS stores data [1]. SQL queries data [2]."

    Returns:
        ["DBMS stores data [1].", "SQL queries data [2]."]
    """

    if not answer or not answer.strip():
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)

    return [
        part.strip()
        for part in parts
        if len(part.strip()) > 15
    ]


def _empty(answer):
    """
    Return a standard result when verification cannot be performed.
    """

    return {
        "kept_answer": answer,
        "claims": [],
        "faithfulness": None,
        "citation_precision": None,
        "verified": False,
    }


def _get_evidence_text(chunk):
    """
    Extract evidence from the document format used by our retrieval.py.

    Our current retrieval.py returns:
        {
            "id": "...",
            "text": "...",
            "source": "...",
            "score": ...
        }

    The original PDF's retrieval module uses a different chunk structure,
    so this helper supports both formats.
    """

    text = chunk.get("parent") or chunk.get("text") or ""
    source = chunk.get("ref") or chunk.get("source") or "Unknown source"

    return f"Source: {source}\n{text[:1200]}"


def verify_answer(answer, chunks, provider=None, max_claims=10):
    """
    Check answer claims against the retrieved evidence.

    Parameters:
        answer: Generated answer text.
        chunks: Retrieved notes/documents.
        provider: Optional LLM provider, such as "gemini".
        max_claims: Maximum number of claims to verify.

    Returns:
        Dictionary containing:
        - kept_answer
        - claims
        - faithfulness
        - citation_precision
        - verified
    """

    claims = split_claims(answer)

    # Nothing to verify if sources or answer claims are missing.
    if (
        not chunks
        or not claims
        or answer.strip().startswith(NOT_FOUND[:9])
    ):
        return _empty(answer)

    # Limit the number of claims to control API usage.
    checked = claims[:max_claims]

    blocks = []
    cited_lists = []

    # Prepare evidence for each claim.
    for index, claim in enumerate(checked, start=1):

        # Find valid citation numbers in this claim.
        cited = sorted({
            int(number)
            for number in CITE.findall(claim)
            if 1 <= int(number) <= len(chunks)
        })

        cited_lists.append(cited)

        # If the claim has valid citations, use those sources.
        # Otherwise, use all available sources for comparison.
        source_numbers = (
            cited if cited
            else list(range(1, len(chunks) + 1))
        )

        evidence_parts = []

        for number in source_numbers:
            evidence = _get_evidence_text(chunks[number - 1])
            evidence_parts.append(f"[{number}] {evidence}")

        evidence_text = "\n\n".join(evidence_parts)

        blocks.append(
            f"Claim {index}: {claim}\n"
            f"Evidence for claim {index}:\n{evidence_text}"
        )

    # Ask the LLM to judge whether each claim is supported.
    prompt = (
        "You are a careful evidence-verification judge.\n"
        "For every claim, decide whether the supplied evidence "
        "fully supports the claim.\n\n"
        "Return a JSON list only. Each item must contain:\n"
        '- "id": the claim number\n'
        '- "supported": true or false\n\n'
        "Example response:\n"
        '[{"id": 1, "supported": true}, '
        '{"id": 2, "supported": false}]\n\n'
        "Do not add explanations outside the JSON list.\n\n"
        + "\n\n".join(blocks)
    )

    verdicts = ask_json(
        prompt,
        provider=provider,
        cache=True,
    )

    # Convert the LLM response into a claim-number lookup.
    support = {}

    if isinstance(verdicts, list):
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue

            if "id" not in verdict or "supported" not in verdict:
                continue

            value = verdict["supported"]

            # Accept only actual Boolean verdicts.
            if not isinstance(value, bool):
                continue

            try:
                support[int(verdict["id"])] = value
            except (TypeError, ValueError):
                continue

    # Do not claim verification succeeded if no usable verdict exists.
    if not support:
        return _empty(answer)

    # Build the final claim report.
    rows = []

    for index, (claim, cited) in enumerate(
        zip(checked, cited_lists),
        start=1,
    ):
        rows.append({
            "text": claim,
            "cited": cited,
            "supported": support.get(index, False),
        })

    # Keep supported claims and any claims beyond the verification limit.
    kept_claims = [
        row["text"]
        for row in rows
        if row["supported"]
    ]

    kept_claims.extend(claims[max_claims:])

    # Citation precision considers claims that actually have citations.
    cited_rows = [
        row for row in rows
        if row["cited"]
    ]

    faithfulness = (
        sum(row["supported"] for row in rows) / len(rows)
        if rows else None
    )

    citation_precision = (
        sum(row["supported"] for row in cited_rows) / len(cited_rows)
        if cited_rows else None
    )

    return {
        "kept_answer": (
            "\n".join(kept_claims)
            if kept_claims
            else NOT_FOUND
        ),
        "claims": rows,
        "faithfulness": faithfulness,
        "citation_precision": citation_precision,
        "verified": True,
    }


if __name__ == "__main__":

    # Simple local test using the notes format in our project.
    sample_answer = (
        "Normalization reduces data redundancy [1]. "
        "Normalization makes every computer run faster [1]."
    )

    sample_chunks = [
        {
            "id": "DBMS_1",
            "source": "DBMS.txt",
            "text": "Normalization reduces data redundancy.",
            "score": 1.2,
        }
    ]

    print("Original answer:")
    print(sample_answer)

    print("\nVerification is starting...")
    result = verify_answer(sample_answer, sample_chunks)

    print("\nKept answer:")
    print(result["kept_answer"])

    print("\nFaithfulness:", result["faithfulness"])
    print("Citation precision:", result["citation_precision"])
    print("Verified:", result["verified"])

    print("\nClaim details:")
    for claim in result["claims"]:
        print(claim)
