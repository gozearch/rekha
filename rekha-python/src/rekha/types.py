from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TypedDict


class ConsistencyLevel(Enum):
    ONE = 1
    QUORUM = 2
    ALL = 3


class GetResult(TypedDict, total=False):
    ids: List[int]
    embeddings: Optional[List[List[float]]]
    metadatas: Optional[List[Optional[Dict[str, str]]]]
    documents: Optional[List[Optional[str]]]
    included: List[str]


class QueryResult(TypedDict, total=False):
    ids: List[List[int]]
    distances: Optional[List[List[float]]]
    embeddings: Optional[List[List[float]]]
    metadatas: Optional[List[List[Optional[Dict[str, str]]]]]
    documents: Optional[List[List[Optional[str]]]]
    included: List[str]


@dataclass
class Payload:
    content_type: str
    data: bytes

    @classmethod
    def from_bytes(cls, data: bytes, content_type: str = "raw") -> Payload:
        return cls(content_type=content_type, data=data)


@dataclass
class ScoredPoint:
    id: int
    score: float
    payload: Optional[Payload] = None


@dataclass
class SearchParams:
    ef_search: int = 100
    nprobe: int = 32
    include_payloads: bool = False


@dataclass
class SearchStats:
    total_ms: float = 0.0
    nodes_contacted: int = 0
    vectors_scanned: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class NodeInfo:
    node_id: str
    address: str
    partition_id: int
    dim_groups: List[int]
    storage_bytes: int
    status: str


@dataclass
class ClusterTopology:
    cluster_id: str
    peers: List[NodeInfo]


@dataclass
class CollectionConfig:
    dim: int = 0
    num_vector_shards: int = 1
    replication_factor: int = 1
    num_dim_groups: int = 1
    dim_group_size: int = 0
    nlist: int = 4096
    nprobe: int = 32
    pq_num_sub_vectors: int = 0
    pq_num_centroids: int = 0
    re_rank_k: int = 0


@dataclass
class CollectionInfo:
    name: str
    config: Optional[CollectionConfig] = None
    vector_count: int = 0
    index_ready: bool = False
    config_timestamp: int = 0
