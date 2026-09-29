"""API: session isolation, upload validation, SSE order, TTL, path safety."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from datasleuth.api.app import create_app
from datasleuth.config import Settings
from datasleuth.llm.client import LLM
from datasleuth.llm.providers import FakeProvider

CSV_A = b"region,amount\nnorth,10\nsouth,20\nnorth,5\n"
CSV_B = b"secret_col,other\nx,1\ny,2\n"


def reply(role_sql: str) -> Any:
    def handler(system: str, user: str) -> str:
        if "planning stage" in system:
            return json.dumps(
                {"status": "answerable", "question_types": ["aggregation"], "steps": "s"}
            )
        if "careful data analyst" in system:
            return json.dumps({"language": "sql", "code": role_sql, "rationale": "r"})
        return json.dumps({"answer": "The total is 35."})

    return handler


def make_client(
    tmp_path: Path, sql: str = "SELECT SUM(amount) AS total FROM a", **kw: Any
) -> TestClient:
    settings = Settings(
        llm_provider="fake", data_dir=tmp_path / "data", cache_dir=tmp_path / "c", log_dir=tmp_path / "l",
        llm_rpm=0, use_cache=False, use_chart=False, schema_linker_min_tables=99,
        frontend_dist=tmp_path / "nofrontend", **kw,
    )  # fmt: skip
    llm = LLM(settings, FakeProvider(reply(sql)), sleep=lambda _s: None)
    return TestClient(create_app(settings, llm=llm))


def new_session(client: TestClient) -> str:
    return client.post("/sessions").json()["session_id"]


def upload(client: TestClient, sid: str, name: str, data: bytes):
    return client.post(
        f"/sessions/{sid}/upload", files={"files": (name, io.BytesIO(data), "text/csv")}
    )


def parse_sse(text: str) -> list[tuple[str, Any]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_happy_path_upload_schema_preview_ask_trace(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    up = upload(client, sid, "a.csv", CSV_A)
    assert up.status_code == 200 and up.json()["tables"][0]["name"] == "a"
    assert client.get(f"/sessions/{sid}/schema").json()["tables"][0]["row_count"] == 3
    prev = client.get(f"/sessions/{sid}/preview/a").json()
    assert prev["columns"] == ["region", "amount"] and len(prev["rows"]) == 3
    res = client.post(f"/sessions/{sid}/ask", json={"question": "total?"})
    events = parse_sse(res.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "result"
    progress = [d["node"] for e, d in events if e == "progress" and d["type"] == "node_start"]
    assert progress == ["ingest", "plan", "link", "write", "validate", "execute", "critic",
                        "analyst", "chart", "report"]  # fmt: skip
    report = events[-1][1]
    assert report["status"] == "success" and report["rows"] == [[35.0]]
    trace = client.get(f"/traces/{report['trace_id']}")
    assert trace.status_code == 200 and trace.json()["answer"] == "The total is 35."
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}/schema").status_code == 404


def test_sessions_are_isolated(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    a, b = new_session(client), new_session(client)
    upload(client, a, "a.csv", CSV_A)
    upload(client, b, "b.csv", CSV_B)
    assert [t["name"] for t in client.get(f"/sessions/{a}/schema").json()["tables"]] == ["a"]
    assert [t["name"] for t in client.get(f"/sessions/{b}/schema").json()["tables"]] == ["b"]
    assert client.get(f"/sessions/{b}/preview/a").status_code == 404  # A's table invisible to B
    # B's agent runs against B's database: A's table does not exist there.
    res = client.post(f"/sessions/{b}/ask", json={"question": "total?"})
    report = parse_sse(res.text)[-1][1]
    assert report["status"] == "failed"
    assert all(a_["result"]["error_type"] == "unknown_table" for a_ in report["attempts"][:1])
    assert "secret_col" not in json.dumps(client.get(f"/sessions/{a}/schema").json())


def test_delete_removes_data_from_disk(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    upload(client, sid, "a.csv", CSV_A)
    directory = tmp_path / "data" / "sessions" / sid
    assert (directory / "data.sqlite").exists()
    client.delete(f"/sessions/{sid}")
    assert not directory.exists()


@pytest.mark.parametrize(
    ("name", "data", "status", "msg"),
    [
        ("evil.exe", b"MZ\x90", 400, "Unsupported"),
        ("empty.csv", b"", 400, "empty"),
        ("bin.csv", b"a\x00b\x00" * 50, 400, "binary"),
        ("fake.xlsx", b"not a zip at all", 400, "XLSX"),
    ],
)
def test_upload_validation(tmp_path: Path, name: str, data: bytes, status: int, msg: str) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    res = upload(client, sid, name, data)
    assert res.status_code == status and msg in res.json()["detail"]
    assert client.get(f"/sessions/{sid}/schema").status_code == 404  # nothing was stored


def test_upload_size_limit_413(tmp_path: Path) -> None:
    client = make_client(tmp_path, upload_max_mb=1)
    sid = new_session(client)
    res = upload(client, sid, "big.csv", b"a,b\n" + b"1,2\n" * 400_000)
    assert res.status_code == 413 and "1 MB" in res.json()["detail"]


def test_failed_second_upload_keeps_first_data(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    upload(client, sid, "a.csv", CSV_A)
    assert upload(client, sid, "bad.exe", b"MZ").status_code == 400
    tables = client.get(f"/sessions/{sid}/schema").json()["tables"]
    assert [t["name"] for t in tables] == ["a"]


def test_second_upload_adds_tables(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    upload(client, sid, "a.csv", CSV_A)
    upload(client, sid, "b.csv", CSV_B)
    tables = client.get(f"/sessions/{sid}/schema").json()["tables"]
    assert sorted(t["name"] for t in tables) == ["a", "b"]


@pytest.mark.parametrize("bad_id", ["../etc", "..%2f..%2fetc", "A" * 32, "zz", "0" * 31])
def test_malformed_session_ids_never_touch_the_filesystem(tmp_path: Path, bad_id: str) -> None:
    client = make_client(tmp_path)
    assert client.get(f"/sessions/{bad_id}/schema").status_code == 404
    assert client.delete(f"/sessions/{bad_id}").status_code in (404, 405)


def test_trace_lookup_rejects_traversal(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/traces/..%2f..%2fsessions").status_code in (404, 422)
    assert client.get("/traces/" + "0" * 32).status_code == 404


def test_no_server_paths_leak(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    bodies = [upload(client, sid, "a.csv", CSV_A).text, client.get(f"/sessions/{sid}/schema").text,
              client.get(f"/sessions/{sid}/preview/a").text, upload(client, sid, "x.exe", b"MZ").text]  # fmt: skip
    events = client.post(f"/sessions/{sid}/ask", json={"question": "total?"}).text
    trace_id = parse_sse(events)[-1][1]["trace_id"]
    bodies += [events, client.get(f"/traces/{trace_id}").text]
    for body in bodies:
        assert str(tmp_path) not in body and "data.sqlite" not in body and ".cache" not in body


def test_expired_sessions_are_cleaned(tmp_path: Path) -> None:
    client = make_client(tmp_path, session_ttl_min=0)
    sid = new_session(client)
    time.sleep(0.05)
    assert client.app.state.store.cleanup_expired() == 1  # type: ignore[attr-defined]
    assert client.get(f"/sessions/{sid}/schema").status_code == 404


def test_ask_requires_data_and_valid_question(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    assert client.post(f"/sessions/{sid}/ask", json={"question": "hi"}).status_code == 409
    upload(client, sid, "a.csv", CSV_A)
    assert client.post(f"/sessions/{sid}/ask", json={"question": ""}).status_code == 422
    assert client.post(f"/sessions/{sid}/ask", json={"question": "x" * 2001}).status_code == 422


def test_connect_disabled_by_default_and_sqlite_refused(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    sid = new_session(client)
    assert (
        client.post(f"/sessions/{sid}/connect", json={"url": "postgresql://u:p@h/db"}).status_code
        == 403
    )
    enabled = make_client(tmp_path / "e", allow_db_connect=True)
    sid2 = new_session(enabled)
    res = enabled.post(f"/sessions/{sid2}/connect", json={"url": "sqlite:///C:/secrets.db"})
    assert res.status_code == 400
    bad = enabled.post(f"/sessions/{sid2}/connect", json={"url": "oracle://u:p@h/db"})
    assert bad.status_code == 400 and "p@h" not in bad.text


def test_llm_errors_become_friendly_sse_error(tmp_path: Path) -> None:
    client = make_client(tmp_path, llm_daily_limit=1)
    sid = new_session(client)
    upload(client, sid, "a.csv", CSV_A)
    events = parse_sse(client.post(f"/sessions/{sid}/ask", json={"question": "q"}).text)
    assert events[-1][0] == "error" and "quota" in events[-1][1]["message"].lower()


def test_cors_and_security_headers(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    res = client.options(
        "/sessions",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert client.get("/healthz").headers["x-content-type-options"] == "nosniff"


def test_openapi_schema_lists_all_routes(tmp_path: Path) -> None:
    paths = make_client(tmp_path).get("/openapi.json").json()["paths"]
    for route in ("/sessions", "/sessions/{session_id}/upload", "/sessions/{session_id}/schema",
                  "/sessions/{session_id}/ask", "/traces/{trace_id}", "/sessions/{session_id}"):  # fmt: skip
        assert route in paths
