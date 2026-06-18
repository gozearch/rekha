from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import grpc
import pytest

from rekha import (
    RekhaClient,
    RekhaConnectError,
    RekhaError,
    RekhaRequestError,
    ScoredPoint,
    SearchParams,
    SearchStats,
)
from rekha.types import ClusterTopology, NodeInfo, Payload


class FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str):
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


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

    def Insert(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Insert", request, timeout=timeout)

    def Search(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Search", request, timeout=timeout)

    def Delete(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Delete", request, timeout=timeout)

    def Fetch(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Fetch", request, timeout=timeout)

    def Handshake(self, request: Any, timeout: float = 0) -> Any:
        return self._call("Handshake", request, timeout=timeout)


@dataclass
class MockStats:
    total_ms: float = 0.0
    nodes_contacted: int = 0
    vectors_scanned: int = 0
    warnings: List[str] = field(default_factory=list)


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


class FakeResponse:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestRekhaClient:
    def test_connect_empty_seeds(self) -> None:
        with pytest.raises(RekhaError, match="at least one seed node"):
            RekhaClient(seeds=[])

    def test_insert_ok(self) -> None:
        stub = FakeStub({
            "Insert": lambda request, timeout: FakeResponse(success=True, error="", id=0),
        })
        client = make_client(stub)
        result = client.insert([0.1, 0.2, 0.3], "default", id=42)
        assert result == 0

    def test_insert_default_id(self) -> None:
        stub = FakeStub({
            "Insert": lambda request, timeout: FakeResponse(success=True, error="", id=0),
        })
        client = make_client(stub)
        result = client.insert([0.1, 0.2, 0.3], "default")
        assert result == 0

    def test_insert_with_payload(self) -> None:
        stub = FakeStub({
            "Insert": lambda request, timeout: FakeResponse(success=True, error="", id=0),
        })
        client = make_client(stub)
        client.insert([0.1, 0.2, 0.3], "default", payload=b'{"k":"v"}')

    def test_insert_retry_then_succeed(self) -> None:
        stub = FakeStub({
            "Insert": lambda request, timeout: FakeResponse(success=True, error="", id=0),
        })
        stub.set_fails("Insert", 1)
        client = make_client(stub)
        client.insert([0.1, 0.2, 0.3], "default")
        assert stub._call_counts.get("Insert", 0) == 2

    def test_search_ok(self) -> None:
        point = FakeResponse(id=1, score=0.5, payload=None)
        stub = FakeStub({
            "Search": lambda request, timeout: FakeResponse(
                results=[point],
                stats=MockStats(),
            ),
        })
        client = make_client(stub)
        results = client.search([0.1, 0.2, 0.3], 10, "default")
        assert len(results) == 1
        assert results[0].id == 1
        assert results[0].score == 0.5

    def test_search_with_params(self) -> None:
        point = FakeResponse(id=1, score=0.5, payload=None)
        stub = FakeStub({
            "Search": lambda request, timeout: FakeResponse(
                results=[point],
                stats=MockStats(total_ms=1.5, nodes_contacted=2, vectors_scanned=100),
            ),
        })
        client = make_client(stub)
        results, stats = client.search_with_params(
            [0.1, 0.2, 0.3], 10, "default", SearchParams(ef_search=200)
        )
        assert len(results) == 1
        assert stats.total_ms == 1.5
        assert stats.nodes_contacted == 2

    def test_delete_ok(self) -> None:
        stub = FakeStub({
            "Delete": lambda request, timeout: FakeResponse(deleted_count=3),
        })
        client = make_client(stub)
        count = client.delete([1, 2, 3], "default")
        assert count == 3

    def test_fetch_ok(self) -> None:
        point = FakeResponse(id=1, score=0.5, payload=None)
        stub = FakeStub({
            "Fetch": lambda request, timeout: FakeResponse(points=[point]),
        })
        client = make_client(stub)
        results = client.fetch([1], "default", include_payloads=True)
        assert len(results) == 1
        assert results[0].id == 1

    def test_fetch_with_payload(self) -> None:
        pb_payload = FakeResponse(content_type="json", data=b'{"k":"v"}')
        point = FakeResponse(id=1, score=0.5, payload=pb_payload)
        stub = FakeStub({
            "Fetch": lambda request, timeout: FakeResponse(points=[point]),
        })
        client = make_client(stub)
        results = client.fetch([1], "default", include_payloads=True)
        assert len(results) == 1
        assert results[0].payload is not None
        assert results[0].payload.data == b'{"k":"v"}'

    def test_cluster_info_ok(self) -> None:
        peer = FakeResponse(
            node_id="n1", address="localhost:50051", partition_id=0,
            dim_groups=[], is_leader=True, raft_term=1, commit_index=5,
            storage_bytes=100, status="healthy",
        )
        stub = FakeStub({
            "Handshake": lambda request, timeout: FakeResponse(
                cluster_id="test-cluster", peers=[peer], error="",
            ),
        })
        client = make_client(stub)
        info = client.cluster_info()
        assert info.cluster_id == "test-cluster"
        assert len(info.peers) == 1
        assert info.peers[0].node_id == "n1"
        assert info.peers[0].is_leader

    def test_retry_exhausted_raises_error(self) -> None:
        stub = FakeStub({
            "Search": lambda request, timeout: FakeResponse(
                results=[FakeResponse(id=1, score=0.5, payload=None)],
                stats=MockStats(),
            ),
        })
        stub.set_fails("Search", 99)
        client = make_client(stub)
        with pytest.raises(RekhaRequestError):
            client.search([0.1], 5, "default")

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

    def test_search_with_payloads_in_results(self) -> None:
        pb_payload = FakeResponse(content_type="json", data=b'{"k":"v"}')
        point = FakeResponse(id=1, score=0.5, payload=pb_payload)
        stub = FakeStub({
            "Search": lambda request, timeout: FakeResponse(
                results=[point],
                stats=MockStats(),
            ),
        })
        client = make_client(stub)
        results, _ = client.search_with_params(
            [0.1, 0.2, 0.3], 10, "default", SearchParams(include_payloads=True)
        )
        assert results[0].payload is not None
        assert results[0].payload.data == b'{"k":"v"}'
