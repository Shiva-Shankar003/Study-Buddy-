from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHILD_SIZE,
    chunk_overlap=config.CHILD_OVERLAP
)

def make_chunks(units, doc, subject, kind):
    rows = []
    for u in units:
        parent = u["text"][:config.PARENT_MAX]
        for piece in _splitter.split_text(u["text"]):
            if len(piece.strip()) < 20:
                continue
            rows.append({
                "doc": doc,
                "subject": subject,
                "kind": kind,
                "loc": u["loc"],
                "ref": u["ref"],
                "text": piece,
                "parent": parent
            })
    return rows