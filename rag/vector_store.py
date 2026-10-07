"""Qdrant vector store: one collection holding a dense vector AND a BM25 sparse vector per chunk.

Modes (set in .env):
  - Local, no server:  QDRANT_PATH=data/vector_store        (embedded, file-based)
  - Docker / Cloud:    QDRANT_URL=http://localhost:6333     (+ QDRANT_API_KEY for Qdrant Cloud)
"""
import uuid

from qdrant_client import QdrantClient, models

from config import settings

COLLECTION = settings.QDRANT_COLLECTION
DENSE = "dense"
SPARSE = "sparse"


def get_client() -> QdrantClient:
    if settings.QDRANT_URL:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return QdrantClient(path=settings.QDRANT_PATH)


def ensure_collection(client: QdrantClient, dense_dim: int, recreate: bool = False) -> None:
    """Create the collection with named dense + sparse (IDF/BM25) vectors and a payload index on domain."""
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("domain", "doc_id", "chunk_type"):
        client.create_payload_index(COLLECTION, field_name=field,
                                    field_schema=models.PayloadSchemaType.KEYWORD)


def upsert_chunks(client: QdrantClient, chunks: list[dict], dense_vecs: list[list[float]],
                  sparse_vecs: list, batch_size: int = 128) -> int:
    """Upsert chunks with their dense + sparse vectors. The whole chunk dict becomes the payload.

    Point IDs are deterministic (source + chunk_index), so re-running ingestion overwrites
    rather than duplicates.
    """
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['source']}#{c.get('qa_id', '')}#{c['chunk_index']}")),
            vector={
                DENSE: d,
                SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
            },
            payload=c,
        )
        for c, d, s in zip(chunks, dense_vecs, sparse_vecs, strict=True)
    ]
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION, points=points[i:i + batch_size], wait=True)
    return len(points)
