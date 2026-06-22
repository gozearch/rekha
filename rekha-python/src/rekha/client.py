from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional

import grpc

import rekha.proto.rekha_pb2 as pb
import rekha.proto.rekha_pb2_grpc as pb_grpc
from .collection import Collection
from .errors import RekhaConnectError, RekhaError, RekhaRequestError
from .types import CollectionConfig, CollectionInfo, ConsistencyLevel


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
    ) -> tuple[grpc.Channel, pb_grpc.RekhaStub]:
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

    def _pb_config(self, **kwargs) -> pb.CollectionConfig:
        return pb.CollectionConfig(
            dim=kwargs.get("dim", 0),
            num_vector_shards=kwargs.get("num_vector_shards", 1),
            replication_factor=kwargs.get("replication_factor", 1),
            num_dim_groups=kwargs.get("num_dim_groups", 1),
            dim_group_size=kwargs.get("dim_group_size", 0),
            nlist=kwargs.get("nlist", 4096),
            nprobe=kwargs.get("nprobe", 32),
            pq_num_sub_vectors=kwargs.get("pq_num_sub_vectors", 0),
            pq_num_centroids=kwargs.get("pq_num_centroids", 0),
            re_rank_k=kwargs.get("re_rank_k", 0),
        )

    def _collection_info_from_pb(self, info_pb) -> CollectionInfo:
        cfg = info_pb.config
        return CollectionInfo(
            name=info_pb.name,
            config=CollectionConfig(
                dim=cfg.dim,
                num_vector_shards=cfg.num_vector_shards,
                replication_factor=cfg.replication_factor,
                num_dim_groups=cfg.num_dim_groups,
                dim_group_size=cfg.dim_group_size,
                nlist=cfg.nlist,
                nprobe=cfg.nprobe,
                pq_num_sub_vectors=cfg.pq_num_sub_vectors,
                pq_num_centroids=cfg.pq_num_centroids,
                re_rank_k=cfg.re_rank_k,
            ) if cfg else None,
            vector_count=info_pb.vector_count,
            index_ready=info_pb.index_ready,
            config_timestamp=getattr(info_pb, 'config_timestamp', 0),
        )

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        get_or_create: bool = False,
        consistency: ConsistencyLevel | None = None,
        **config: Any,
    ) -> Collection:
        if get_or_create:
            try:
                return self.get_collection(name)
            except RekhaError:
                pass

        pb_cfg = self._pb_config(**config)
        cl_val = consistency.value if consistency else 0
        request = pb.CreateCollectionRequest(name=name, config=pb_cfg, timestamp=0, consistency=cl_val)
        self._with_retry(
            "create_collection",
            lambda: self._stub.CreateCollection(
                request, timeout=self._config["request_timeout"]
            ),
        )
        info_pb = self._with_retry(
            "list_collections",
            lambda: self._stub.ListCollections(
                pb.ListCollectionsRequest(), timeout=self._config["request_timeout"]
            ),
        )
        coll_info = None
        for ci in info_pb.collections:
            if ci.name == name:
                coll_info = self._collection_info_from_pb(ci)
                break
        return Collection(self, name, info=coll_info)

    def get_collection(self, name: str) -> Collection:
        response = self._with_retry(
            "collection_exists",
            lambda: self._stub.CollectionExists(
                pb.CollectionExistsRequest(name=name),
                timeout=self._config["request_timeout"],
            ),
        )
        if not response.exists:
            raise RekhaError(f"collection '{name}' not found")

        info_pb = self._with_retry(
            "list_collections",
            lambda: self._stub.ListCollections(
                pb.ListCollectionsRequest(), timeout=self._config["request_timeout"]
            ),
        )
        coll_info = None
        for ci in info_pb.collections:
            if ci.name == name:
                coll_info = self._collection_info_from_pb(ci)
                break
        return Collection(self, name, info=coll_info)

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        consistency: ConsistencyLevel | None = None,
        **config: Any,
    ) -> Collection:
        return self.create_collection(name, metadata=metadata, get_or_create=True, consistency=consistency, **config)

    def delete_collection(self, name: str, consistency: ConsistencyLevel | None = None) -> None:
        cl_val = consistency.value if consistency else 0
        self._with_retry(
            "delete_collection",
            lambda: self._stub.DropCollection(
                pb.DropCollectionRequest(name=name, timestamp=0, consistency=cl_val),
                timeout=self._config["request_timeout"],
            ),
        )

    def list_collections(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Collection]:
        response = self._with_retry(
            "list_collections",
            lambda: self._stub.ListCollections(
                pb.ListCollectionsRequest(), timeout=self._config["request_timeout"]
            ),
        )
        collections = [
            Collection(self, ci.name, info=self._collection_info_from_pb(ci))
            for ci in response.collections
        ]
        if offset is not None:
            collections = collections[offset:]
        if limit is not None:
            collections = collections[:limit]
        return collections

    def count_collections(self) -> int:
        return len(self.list_collections())

    def heartbeat(self) -> bool:
        response = self._with_retry(
            "heartbeat",
            lambda: self._stub.Heartbeat(
                pb.HeartbeatRequest(node_id="", address="", storage_bytes=0),
                timeout=self._config["request_timeout"],
            ),
        )
        return response.success

    def close(self) -> None:
        try:
            self._channel.close()
        except Exception:
            pass

    def __enter__(self) -> RekhaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
