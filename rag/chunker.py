"""Chunking strategy.

Policy documents: token-based recursive splitting, CHUNK_SIZE=500 tokens with CHUNK_OVERLAP=50,
measured with the same tokenizer family as OpenAI embedding models (cl100k_base). The splitter
tries Markdown section breaks ("## ") first, then paragraphs, lines and words, so chunks end at
natural boundaries and a section is only cut mid-way when it alone exceeds 500 tokens. The 50-token
overlap keeps a step or table row that straddles a boundary retrievable from either side.

Each chunk is prefixed with its document title so short sections ("## Lost or stolen phone")
still carry the context of which policy they belong to.

Q&A pairs are short and self-contained, so each one is a single chunk (no splitting).
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
ENCODING = "cl100k_base"

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name=ENCODING,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
)


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split policy docs into chunks; returns [{'text', **metadata, 'chunk_index'}]."""
    chunks = []
    for doc in docs:
        meta = doc["metadata"]
        pieces = _splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece.startswith("# "):
                piece = f"# {meta['title']}\n\n{piece}"
            chunks.append({**meta, "text": piece, "chunk_index": i, "n_chunks": len(pieces)})
    return chunks


def qa_to_chunks(records: list[dict]) -> list[dict]:
    """Q&A pairs map 1:1 to chunks."""
    return [{**r["metadata"], "text": r["text"], "chunk_index": 0, "n_chunks": 1} for r in records]


def count_tokens(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding(ENCODING).encode(text))
