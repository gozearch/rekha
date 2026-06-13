from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Payload:
    content_type: str  # "json", "text", "raw"
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
    beam_width: int = 4
    include_payloads: bool = False
    partition_hint: Optional[int] = None


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
    is_leader: bool
    raft_term: int
    commit_index: int
    storage_bytes: int
    status: str


@dataclass
class ClusterTopology:
    cluster_id: str
    peers: List[NodeInfo]
