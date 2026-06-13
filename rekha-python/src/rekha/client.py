from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple

import grpc

import rekha.proto.rekha_pb2 as pb
import rekha.proto.rekha_pb2_grpc as pb_grpc
from .errors import RekhaConnectError, RekhaError, RekhaRequestError
from .types import ClusterTopology, NodeInfo, Payload, ScoredPoint, SearchParams, SearchStats


class RekhaClient:
    def __init__(
        self,
        seeds: List[str],
        connect_timeout: float = 10.0,
        request_timeout: float = 60.0,
        max_retries: int = 3,
        use_tls: bool = False,
        ca_cert: Optional[bytes] = None,
    ):
        self._config = {
            "connect_timeout": connect_timeout,
            "request_timeout": request_timeout,
            "max_retries": max_retries,
            "use_tls": use_tls,
            "ca_cert": ca_cert,
        }
        self._channel, self._stub = self._connect(seeds)

    @classmethod
    def connect(
        cls,
        seeds: List[str],
        connect_timeout: float = 10.0,
        request_timeout: float = 60.0,
        max_retries: int = 3,
    ) -> RekhaClient:
        return cls(
            seeds=seeds,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )

    def _connect(
        self, seeds: List[str]
    ) -> Tuple[grpc.Channel, pb_grpc.RekhaStub]:
        if not seeds:
            raise RekhaError("at least one seed node required")

        last_err: Optional[Exception] = None
        for seed in seeds:
            try:
                if self._config["use_tls"]:
                    if self._config["ca_cert"]:
                        creds = grpc.ssl_channel_credentials(
                            root_certificates=self._config["ca_cert"]
                        )
                    else:
                        creds = grpc.ssl_channel_credentials()
                    channel = grpc.secure_channel(seed, creds, options=[
                        ("grpc.connect_timeout_ms", int(self._config["connect_timeout"] * 1000)),
                    ])
                else:
                    channel = grpc.insecure_channel(seed, options=[
                        ("grpc.connect_timeout_ms", int(self._config["connect_timeout"] * 1000)),
                    ])

                grpc.channel_ready_future(channel).result(
                    timeout=self._config["connect_timeout"]
                )
                stub = pb_grpc.RekhaStub(channel)
                return channel, stub
            except Exception as e:
                last_err = e
                continue

        raise RekhaConnectError(
            seed=seeds[-1],
            detail=str(last_err) if last_err else "all seeds failed",
        )

    def _with_retry(self, operation: str, call_fn):
        max_attempts = self._config["max_retries"] + 1
        for attempt in range(max_attempts):
            try:
                return call_fn()
            except grpc.RpcError as e:
                if attempt == max_attempts - 1:
                    raise RekhaRequestError(
                        operation=operation,
                        status_code=e.code().name,
                        detail=e.details() if e.details() else str(e),
                    ) from e
                base_ms = 100 * (2 ** attempt)
                jitter = random.randint(0, base_ms // 4)
                time.sleep((base_ms + jitter) / 1000.0)

    def insert(
        self, id: int, vector: List[float], payload: Optional[bytes] = None
    ) -> None:
        pb_payload = None
        if payload is not None:
            pb_payload = pb.Payload(content_type="raw", data=payload)

        request = pb.InsertRequest(id=id, vector=vector, payload=pb_payload)

        def call():
            response = self._stub.Insert(request, timeout=self._config["request_timeout"])
            if not response.success:
                raise grpc.RpcError(
                    grpc.StatusCode.INTERNAL,
                    response.error if response.error else "insert failed",
                )

        self._with_retry("insert", call)

    def search(
        self, query: List[float], top_k: int
    ) -> List[ScoredPoint]:
        results, _ = self.search_with_params(query, top_k, SearchParams())
        return results

    def search_with_params(
        self,
        query: List[float],
        top_k: int,
        params: SearchParams,
    ) -> Tuple[List[ScoredPoint], SearchStats]:
        pb_params = pb.SearchParams(
            ef_search=params.ef_search,
            beam_width=params.beam_width,
            include_payloads=params.include_payloads,
            partition_hint=params.partition_hint,
        )
        request = pb.SearchRequest(query_vector=query, top_k=top_k, params=pb_params)

        def call():
            response = self._stub.Search(request, timeout=self._config["request_timeout"])
            results = [
                ScoredPoint(
                    id=p.id,
                    score=p.score,
                    payload=Payload(content_type=p.payload.content_type, data=p.payload.data)
                    if p.payload
                    else None,
                )
                for p in response.results
            ]
            stats = SearchStats(
                total_ms=response.stats.total_ms,
                nodes_contacted=response.stats.nodes_contacted,
                vectors_scanned=response.stats.vectors_scanned,
                warnings=list(response.stats.warnings),
            )
            return results, stats

        return self._with_retry("search", call)

    def delete(self, ids: List[int]) -> int:
        request = pb.DeleteRequest(ids=ids)

        def call():
            response = self._stub.Delete(request, timeout=self._config["request_timeout"])
            return response.deleted_count

        return self._with_retry("delete", call)

    def fetch(
        self, ids: List[int], include_payloads: bool = False
    ) -> List[ScoredPoint]:
        request = pb.FetchRequest(ids=ids, include_payloads=include_payloads)

        def call():
            response = self._stub.Fetch(request, timeout=self._config["request_timeout"])
            return [
                ScoredPoint(
                    id=p.id,
                    score=p.score,
                    payload=Payload(content_type=p.payload.content_type, data=p.payload.data)
                    if p.payload
                    else None,
                )
                for p in response.points
            ]

        return self._with_retry("fetch", call)

    def cluster_info(self) -> ClusterTopology:
        request = pb.HandshakeRequest(node_id="", address="")

        def call():
            response = self._stub.Handshake(request, timeout=self._config["request_timeout"])
            peers = [
                NodeInfo(
                    node_id=n.node_id,
                    address=n.address,
                    partition_id=n.partition_id,
                    dim_groups=list(n.dim_groups),
                    is_leader=n.is_leader,
                    raft_term=n.raft_term,
                    commit_index=n.commit_index,
                    storage_bytes=n.storage_bytes,
                    status=n.status,
                )
                for n in response.peers
            ]
            return ClusterTopology(cluster_id=response.cluster_id, peers=peers)

        return self._with_retry("cluster_info", call)

    def search_stream(
        self, query: List[float], top_k: int, params: Optional[SearchParams] = None
    ):
        params = params or SearchParams()
        pb_params = pb.SearchParams(
            ef_search=params.ef_search,
            beam_width=params.beam_width,
            include_payloads=params.include_payloads,
            partition_hint=params.partition_hint,
        )
        request = pb.SearchRequest(query_vector=query, top_k=top_k, params=pb_params)

        for p in self._stub.SearchStream(request, timeout=self._config["request_timeout"]):
            yield ScoredPoint(
                id=p.id,
                score=p.score,
                payload=Payload(content_type=p.payload.content_type, data=p.payload.data)
                if p.payload
                else None,
            )

    def close(self) -> None:
        try:
            self._channel.close()
        except Exception:
            pass

    def __enter__(self) -> RekhaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
