from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import rekha.proto.rekha_pb2 as pb
from .types import ConsistencyLevel, GetResult, QueryResult

if TYPE_CHECKING:
    from .client import RekhaClient


_INCLUDE_DISTANCES = "distances"
_INCLUDE_EMBEDDINGS = "embeddings"
_INCLUDE_METADATAS = "metadatas"
_INCLUDE_DOCUMENTS = "documents"


def _parse_payload(payload_pb: Any) -> tuple[dict | None, str | None]:
    if payload_pb is None:
        return None, None
    try:
        data = payload_pb.data
        ct = payload_pb.content_type
    except AttributeError:
        return None, None
    if ct == "json" and data:
        return json.loads(data), None
    if ct == "text" and data:
        return None, data.decode("utf-8")
    return None, None


class Collection:
    def __init__(self, client: RekhaClient, name: str, info: Any | None = None):
        self._client = client
        self.name = name

    def add(
        self,
        ids: List[int],
        embeddings: List[List[float]],
        metadatas: List[dict | None] | None = None,
        documents: List[str | None] | None = None,
        consistency: ConsistencyLevel | None = None,
    ) -> None:
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids")
        if documents is not None and len(documents) != len(ids):
            raise ValueError("documents length must match ids")

        cl_val = consistency.value if consistency else 0

        def generate():
            for i, (vec_id, vec) in enumerate(zip(ids, embeddings)):
                payload = None
                md = metadatas[i] if metadatas else None
                doc = documents[i] if documents else None
                if md is not None:
                    payload = pb.Payload(content_type="json", data=json.dumps(md).encode())
                elif doc is not None:
                    payload = pb.Payload(content_type="text", data=doc.encode())
                yield pb.InsertRequest(
                    id=vec_id,
                    vector=vec,
                    payload=payload,
                    collection_name=self.name,
                    timestamp=0,
                    consistency=cl_val,
                )

        self._client._with_retry("add", lambda: self._client._stub.InsertBatch(generate()))

    def get(
        self,
        ids: List[int] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: List[str] | None = None,
        consistency: ConsistencyLevel | None = None,
    ) -> GetResult:
        if include is None:
            include = [_INCLUDE_METADATAS, _INCLUDE_DOCUMENTS]

        cl_val = consistency.value if consistency else 0

        if ids is not None:
            response = self._client._with_retry(
                "get",
                lambda: self._client._stub.Fetch(
                    pb.FetchRequest(ids=ids, collection_name=self.name, include_payloads=True, consistency=cl_val),
                    timeout=self._client._config["request_timeout"],
                ),
            )
            all_points = list(response.points)
        else:
            params = pb.SearchParams(ef_search=1000, nprobe=32, include_payloads=True)
            request = pb.SearchRequest(
                query_vector=[0.0],
                top_k=1000000,
                params=params,
                local_only=False,
                collection_name=self.name,
                consistency=cl_val,
            )
            response = self._client._with_retry(
                "get",
                lambda req=request: self._client._stub.Search(
                    req, timeout=self._client._config["request_timeout"]
                ),
            )
            all_points = list(response.results)

        if offset is not None:
            all_points = all_points[offset:]
        if limit is not None:
            all_points = all_points[:limit]

        ids_out: List[int] = []
        embeddings_out: List[List[float]] | None = [] if _INCLUDE_EMBEDDINGS in include else None
        metadatas_out: List[dict | None] | None = [] if _INCLUDE_METADATAS in include else None
        documents_out: List[str | None] | None = [] if _INCLUDE_DOCUMENTS in include else None

        for pt in all_points:
            ids_out.append(pt.id)
            md, doc = _parse_payload(getattr(pt, "payload", None))
            if embeddings_out is not None:
                embeddings_out.append(getattr(pt, "data", []))
            if metadatas_out is not None:
                metadatas_out.append(md)
            if documents_out is not None:
                documents_out.append(doc)

        result: GetResult = {
            "ids": ids_out,
            "included": include,
        }
        if embeddings_out is not None:
            result[_INCLUDE_EMBEDDINGS] = embeddings_out
        if metadatas_out is not None:
            result[_INCLUDE_METADATAS] = metadatas_out
        if documents_out is not None:
            result[_INCLUDE_DOCUMENTS] = documents_out
        return result

    def query(
        self,
        query_embeddings: List[List[float]] | None = None,
        n_results: int = 10,
        include: List[str] | None = None,
        consistency: ConsistencyLevel | None = None,
    ) -> QueryResult:
        if query_embeddings is None or len(query_embeddings) == 0:
            raise ValueError("query_embeddings is required")

        if include is None:
            include = [_INCLUDE_METADATAS, _INCLUDE_DOCUMENTS, _INCLUDE_DISTANCES]

        cl_val = consistency.value if consistency else 0

        batch_ids: List[List[int]] = []
        batch_distances: List[List[float]] = []
        batch_metadatas: List[List[dict | None]] = []
        batch_embeddings: List[List[List[float]]] = []
        batch_documents: List[List[str | None]] = []

        for qvec in query_embeddings:
            pb_params = pb.SearchParams(
                ef_search=100,
                nprobe=32,
                include_payloads=_INCLUDE_METADATAS in include,
            )
            request = pb.SearchRequest(
                query_vector=qvec,
                top_k=n_results,
                params=pb_params,
                local_only=False,
                collection_name=self.name,
                consistency=cl_val,
            )
            response = self._client._with_retry(
                "query",
                lambda req=request: self._client._stub.Search(
                    req, timeout=self._client._config["request_timeout"]
                ),
            )

            point_ids: List[int] = []
            point_dists: List[float] = []
            point_mds: List[dict | None] = []
            point_embs: List[List[float]] = []
            point_docs: List[str | None] = []

            for pt in response.results:
                point_ids.append(pt.id)
                point_dists.append(pt.score)
                md, doc = _parse_payload(getattr(pt, "payload", None))
                point_mds.append(md)
                point_embs.append([])
                point_docs.append(doc)

            batch_ids.append(point_ids)
            batch_distances.append(point_dists)
            batch_metadatas.append(point_mds)
            batch_documents.append(point_docs)

        result: QueryResult = {
            "ids": batch_ids,
            "included": include,
        }
        if _INCLUDE_DISTANCES in include:
            result[_INCLUDE_DISTANCES] = batch_distances
        if _INCLUDE_METADATAS in include:
            result[_INCLUDE_METADATAS] = batch_metadatas
        if _INCLUDE_DOCUMENTS in include:
            result[_INCLUDE_DOCUMENTS] = batch_documents
        if _INCLUDE_EMBEDDINGS in include:
            result[_INCLUDE_EMBEDDINGS] = batch_embeddings
        return result

    def delete(self, ids: List[int], consistency: ConsistencyLevel | None = None) -> None:
        if not ids:
            raise ValueError("ids must not be empty")
        cl_val = consistency.value if consistency else 0
        self._client._with_retry(
            "delete",
            lambda: self._client._stub.Delete(
                pb.DeleteRequest(ids=ids, collection_name=self.name, timestamp=0, consistency=cl_val, is_replication=False),
                timeout=self._client._config["request_timeout"],
            ),
        )
