"""Vector store: Qdrant embedded con named vectors dense + sparse."""
import uuid
from qdrant_client import QdrantClient, models
from app.models import Chunk


def _sparse(svec) -> models.SparseVector:
    # svec.indices / .values possono essere array numpy → conversione a tipi nativi
    return models.SparseVector(
        indices=[int(i) for i in svec.indices],
        values=[float(v) for v in svec.values],
    )


class VectorStore:
    def __init__(self, path: str, collection: str, dense_dim: int) -> None:
        self.client = QdrantClient(path=path)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

    def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse) -> None:
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dvec, "sparse": _sparse(svec)},
                payload=chunk.model_dump(),
            )
            for chunk, dvec, svec in zip(chunks, dense, sparse)
        ]
        self.client.upsert(self.collection, points=points)

    def search_dense(self, vec: list[float], top_k: int) -> list[str]:
        res = self.client.query_points(
            self.collection, query=vec, using="dense", limit=top_k
        )
        return [p.payload["chunk_id"] for p in res.points]

    def search_sparse(self, sparse_vec, top_k: int) -> list[str]:
        res = self.client.query_points(
            self.collection, query=_sparse(sparse_vec), using="sparse", limit=top_k
        )
        return [p.payload["chunk_id"] for p in res.points]

    def get_chunk(self, chunk_id: str) -> Chunk:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_id", match=models.MatchValue(value=chunk_id)
                    )
                ]
            ),
            limit=1,
        )
        return Chunk(**points[0].payload)

    def count(self) -> int:
        return self.client.count(self.collection).count
