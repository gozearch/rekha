from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import grpc
import pytest

from rekha import (
    Collection,
    ConsistencyLevel,
    RekhaClient,
    RekhaConnectError,
    RekhaError,
    RekhaRequestError,
)


class FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str):
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class FakeResponse:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _fake_config(dim: int = 256, **overrides) -> Any:
    cfg = FakeResponse(
        dim=dim,
        num_vector_shards=1,
        replication_factor=1,
        num_dim_groups=1,
        dim_group_size=0,
        nlist=4096,
        nprobe=32,
        pq_num_sub_vectors=0,
        pq_num_centroids=0,
        re_rank_k=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class FakeStub:
    def __init__(self, handlers: Dict[str, Any]):
        self._handlers = handlers
        self._call_counts: Dict[str, int] = {}
        self._fail_counts: Dict[str, int] = {}

    def set_fails(self, method: str, count: int) -> None:
        self._fail_counts[method] = count

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self._call_counts[method] = self._call_counts.get(method, 0) + 1
        fails = self._fail_counts.get(method, 0)
        if self._call_counts[method] <= fails:
            raise FakeRpcError(grpc.StatusCode.UNAVAILABLE, "transient error")
        return self._handlers[method](*args, **kwargs)

    def CreateCollection(self, request: Any, timeout: float = 0) -> Any:
        return self._call("CreateCollection", request, timeout=timeout)

    def DropCollection(self, request: Any, timeout: float = 0) -> Any:
        return self._call("DropCollection", request, timeout=timeout)

    def ListCollections(self, request: Any, timeout: float = 0) -> Any:
        return self._call("ListCollections", request, timeout=timeout)

    def CollectionExists(self, request: Any, timeout: float = 0) -> Any:
        return self._call("CollectionExists", request, timeout=timeout)

    def InsertBatch(self, request_iterator: Any, timeout: float = 0) -> Any:
        self._call_counts["InsertBatch"] = self._call_counts.get("InsertBatch", 0) + 1
        items = list(request_iterator)
        return self._handlers.get("InsertBatch", lambda items: FakeResponse(inserted_count=len(items), errors=[]))(items)

    def Delete(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Delete", request, timeout=timeout)

    def Fetch(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Fetch", request, timeout=timeout)

    def Search(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Search", request, timeout=timeout)


def make_client(stub: Any = None) -> RekhaClient:
    client = RekhaClient.__new__(RekhaClient)
    client._config = {
        "connect_timeout": 10.0,
        "request_timeout": 60.0,
        "max_retries": 3,
        "use_tls": False,
        "ca_cert": None,
    }
    client._channel = None
    client._stub = stub or FakeStub({})
    return client


def make_collection(client: RekhaClient, name: str = "test") -> Collection:
    coll = Collection.__new__(Collection)
    coll._client = client
    coll.name = name
    return coll


def fake_scored_point(id: int, score: float = 0.5, payload: Any = None) -> Any:
    return FakeResponse(id=id, score=score, payload=payload)


def fake_payload(content_type: str, data: bytes) -> Any:
    return FakeResponse(content_type=content_type, data=data)


@dataclass
class MockStats:
    total_ms: float = 0.0
    nodes_contacted: int = 0
    vectors_scanned: int = 0
    warnings: List[str] = field(default_factory=list)


class TestRekhaClient:
    def test_connect_empty_seeds(self) -> None:
        with pytest.raises(RekhaError, match="at least one seed node"):
            RekhaClient(seeds=[])

    def test_default_config(self) -> None:
        client = RekhaClient.__new__(RekhaClient)
        client._config = {
            "connect_timeout": 10.0,
            "request_timeout": 60.0,
            "max_retries": 3,
            "use_tls": False,
            "ca_cert": None,
        }
        assert client._config["max_retries"] == 3

    def test_retry_exhausted_raises_error(self) -> None:
        stub = FakeStub({
            "ListCollections": lambda req, timeout: FakeResponse(collections=[]),
        })
        stub.set_fails("ListCollections", 99)
        client = make_client(stub)
        with pytest.raises(RekhaRequestError):
            client.list_collections()

    def test_create_collection(self) -> None:
        cfg = _fake_config(dim=256, nlist=1024)
        ci = FakeResponse(name="test", config=cfg, vector_count=0, index_ready=False)
        stub = FakeStub({
            "CreateCollection": lambda req, timeout: FakeResponse(success=True, error=""),
            "ListCollections": lambda req, timeout: FakeResponse(collections=[ci]),
        })
        client = make_client(stub)
        coll = client.create_collection("test", dim=256, nlist=1024)
        assert coll.name == "test"

    def test_create_collection_get_or_create_exists(self) -> None:
        cfg = _fake_config(dim=256)
        ci = FakeResponse(name="existing", config=cfg, vector_count=0, index_ready=False)
        stub = FakeStub({
            "CollectionExists": lambda req, timeout: FakeResponse(exists=True),
            "ListCollections": lambda req, timeout: FakeResponse(collections=[ci]),
        })
        client = make_client(stub)
        coll = client.create_collection("existing", get_or_create=True)
        assert coll.name == "existing"

    def test_drop_collection(self) -> None:
        stub = FakeStub({
            "DropCollection": lambda req, timeout: FakeResponse(success=True, error=""),
        })
        client = make_client(stub)
        client.drop_collection("test")
        assert stub._call_counts.get("DropCollection", 0) == 1

    def test_list_collections(self) -> None:
        cfg = _fake_config(dim=256)
        ci_a = FakeResponse(name="a", config=cfg, vector_count=0, index_ready=True)
        ci_b = FakeResponse(name="b", config=cfg, vector_count=0, index_ready=True)
        stub = FakeStub({
            "ListCollections": lambda req, timeout: FakeResponse(collections=[ci_a, ci_b]),
        })
        client = make_client(stub)
        colls = client.list_collections()
        assert len(colls) == 2
        assert colls[0].name == "a"
        assert colls[1].name == "b"

    def test_list_collections_limit_offset(self) -> None:
        cfg = _fake_config(dim=256)
        ci_a = FakeResponse(name="a", config=cfg, vector_count=0, index_ready=True)
        ci_b = FakeResponse(name="b", config=cfg, vector_count=0, index_ready=True)
        ci_c = FakeResponse(name="c", config=cfg, vector_count=0, index_ready=True)
        stub = FakeStub({
            "ListCollections": lambda req, timeout: FakeResponse(collections=[ci_a, ci_b, ci_c]),
        })
        client = make_client(stub)
        colls = client.list_collections(limit=1, offset=1)
        assert len(colls) == 1
        assert colls[0].name == "b"


class TestCollection:
    def test_add(self) -> None:
        stub = FakeStub({
            "InsertBatch": lambda items: FakeResponse(inserted_count=2, errors=[]),
        })
        client = make_client(stub)
        coll = make_collection(client)
        coll.add(ids=[1, 2], embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        assert stub._call_counts.get("InsertBatch", 0) == 1

    def test_add_with_metadata(self) -> None:
        stub = FakeStub({
            "InsertBatch": lambda items: FakeResponse(inserted_count=1, errors=[]),
        })
        client = make_client(stub)
        coll = make_collection(client)
        coll.add(
            ids=[1],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"tag": "test"}],
        )
        assert stub._call_counts.get("InsertBatch", 0) == 1

    def test_add_metadata_length_mismatch(self) -> None:
        stub = FakeStub({})
        client = make_client(stub)
        coll = make_collection(client)
        with pytest.raises(ValueError, match="metadatas length"):
            coll.add(ids=[1, 2], embeddings=[[0.1], [0.2]], metadatas=[{"k": "v"}])

    def test_get_by_ids(self) -> None:
        payload = fake_payload(content_type="json", data=b'{"tag":"cat"}')
        pt = fake_scored_point(id=1, score=0.0, payload=payload)
        stub = FakeStub({
            "Fetch": lambda req, timeout: FakeResponse(points=[pt], vectors=[]),
        })
        client = make_client(stub)
        coll = make_collection(client)
        result = coll.get(ids=[1])
        assert result["ids"] == [1]
        assert result["metadatas"] == [{"tag": "cat"}]

    def test_get_all(self) -> None:
        pt1 = fake_scored_point(id=1, score=0.9)
        pt2 = fake_scored_point(id=2, score=0.8)
        stub = FakeStub({
            "Search": lambda req, timeout: FakeResponse(
                results=[pt1, pt2],
                stats=MockStats(),
            ),
        })
        client = make_client(stub)
        coll = make_collection(client)
        result = coll.get()
        assert result["ids"] == [1, 2]

    def test_get_limit_offset(self) -> None:
        pts = [fake_scored_point(id=i, score=1.0 - i * 0.1) for i in range(5)]
        stub = FakeStub({
            "Search": lambda req, timeout: FakeResponse(
                results=pts, stats=MockStats(),
            ),
        })
        client = make_client(stub)
        coll = make_collection(client)
        result = coll.get(limit=2, offset=1)
        assert result["ids"] == [1, 2]

    def test_get_include_embeddings(self) -> None:
        stub = FakeStub({
            "Fetch": lambda req, timeout: FakeResponse(
                points=[FakeResponse(id=1, score=0.0, payload=None)],
                vectors=[FakeResponse(id=1, data=[0.1, 0.2])],
            ),
        })
        client = make_client(stub)
        coll = make_collection(client)
        result = coll.get(ids=[1], include=["embeddings"])
        assert "embeddings" in result
        assert result["included"] == ["embeddings"]

    def test_query(self) -> None:
        pt1 = fake_scored_point(id=1, score=0.9)
        pt2 = fake_scored_point(id=2, score=0.8)
        stub = FakeStub({
            "Search": lambda req, timeout: FakeResponse(
                results=[pt1, pt2], stats=MockStats(),
            ),
        })
        client = make_client(stub)
        coll = make_collection(client)
        result = coll.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=2)
        assert len(result["ids"]) == 1
        assert result["ids"][0] == [1, 2]
        assert result["distances"][0] == [0.9, 0.8]

    def test_query_empty_embeddings(self) -> None:
        client = make_client(FakeStub({}))
        coll = make_collection(client)
        with pytest.raises(ValueError, match="query_embeddings"):
            coll.query(query_embeddings=[])

    def test_query_multiple_embeddings(self) -> None:
        pt1 = fake_scored_point(id=1, score=0.9)
        pt2 = fake_scored_point(id=10, score=0.85)
        stub = FakeStub({
            "Search": lambda req, timeout: FakeResponse(
                results=[pt1], stats=MockStats(),
            ) if req.query_vector == [0.1, 0.2, 0.3, 0.4] else FakeResponse(
                results=[pt2], stats=MockStats(),
            ),
        })
        client = make_client(stub)
        coll = make_collection(client)
        result = coll.query(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            n_results=1,
        )
        assert len(result["ids"]) == 2

    def test_delete_by_ids(self) -> None:
        stub = FakeStub({
            "Delete": lambda req, timeout: FakeResponse(deleted_count=3),
        })
        client = make_client(stub)
        coll = make_collection(client)
        coll.delete(ids=[1, 2, 3])
        assert stub._call_counts.get("Delete", 0) == 1

    def test_delete_empty_ids(self) -> None:
        client = make_client(FakeStub({}))
        coll = make_collection(client)
        with pytest.raises(ValueError, match="ids must not be empty"):
            coll.delete(ids=[])

    def test_consistency_level_enum_values(self) -> None:
        assert ConsistencyLevel.ONE.value == 1
        assert ConsistencyLevel.QUORUM.value == 2
        assert ConsistencyLevel.ALL.value == 3
