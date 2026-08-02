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
