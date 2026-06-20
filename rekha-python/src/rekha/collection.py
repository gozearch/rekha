from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .types import CollectionMeta, GetResult, QueryResult

if TYPE_CHECKING:
    from .client import RekhaClient


class Collection:
    """A Rekha collection, analogous to a table in ChromaDB.

    Provides add/get/update/upsert/delete/query/count operations
    on a named vector collection.
    """

    def __init__(self, client: RekhaClient, name: str, meta: CollectionMeta):
        self._client = client
        self.name = name
        self._meta = meta

    @property
    def dim(self) -> int:
        return self._meta.dim

    @property
    def vector_count(self) -> int:
        return self._meta.vector_count

    @property
    def index_ready(self) -> bool:
        return self._meta.index_ready

    def add(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Add records to the collection.

        At least one of embeddings or documents must be provided.
        IDs are hashed to u64 internally for storage.
        """
        if embeddings is None and documents is None:
            raise ValueError("At least one of embeddings or documents must be provided")
        if embeddings is not None and len(embeddings) != len(ids):
            raise ValueError(f"embeddings length ({len(embeddings)}) != ids length ({len(ids)})")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(f"metadatas length ({len(metadatas)}) != ids length ({len(ids)})")
        if documents is not None and len(documents) != len(ids):
            raise ValueError(f"documents length ({len(documents)}) != ids length ({len(ids)})")

        for i, id_str in enumerate(ids):
            vector = embeddings[i] if embeddings else None
            payload_bytes = None
            if metadatas or documents:
                meta_dict: Dict[str, Any] = {}
                if metadatas:
                    meta_dict.update(metadatas[i])
                if documents:
                    meta_dict["_document"] = documents[i]
                payload_bytes = self._client._serialize_metadata(meta_dict)
            self._client.insert(
                vector=vector or [0.0] * self._meta.dim,
                collection_name=self.name,
                id=0,
                payload=payload_bytes,
            )

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> GetResult:
        raise NotImplementedError("get not yet implemented via server API")

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError("update not yet implemented via server API")

    def upsert(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError("upsert not yet implemented via server API")

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        raise NotImplementedError("delete with string IDs not yet implemented")

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> QueryResult:
        single = len(query_embeddings) == 1
        all_ids: List[List[str]] = []
        all_distances: List[List[float]] = []

        for qv in query_embeddings:
            results, _ = self._client.search_with_params(
                query=qv,
                top_k=n_results,
                collection_name=self.name,
                params=self._client._default_params(),
            )
            ids = [str(r.id) for r in results]
            dists = [r.score for r in results]
            all_ids.append(ids)
            all_distances.append(dists)

        return QueryResult(
            ids=all_ids,
            distances=all_distances,
        )

    def count(self) -> int:
        raise NotImplementedError("count not yet implemented via server API")

    def peek(self, n: int = 10) -> GetResult:
        raise NotImplementedError("peek not yet implemented via server API")

    def modify(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError("modify not yet implemented via server API")
