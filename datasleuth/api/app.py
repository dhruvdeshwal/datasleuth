"""FastAPI application: sessions, uploads, streamed questions and traces."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from datasleuth.api.sessions import SessionNotFound, SessionStore
from datasleuth.config import Settings, get_settings
from datasleuth.db.connectors import ConnectionRejected
from datasleuth.db.ingest import UploadError
from datasleuth.graph import DataSleuthAgent
from datasleuth.llm.base import LLMError
from datasleuth.llm.client import LLM
from datasleuth.sandbox.executor import Sandbox
from datasleuth.state import Report, SchemaInfo

log = logging.getLogger("datasleuth.api")
_MAX_FILES_PER_UPLOAD = 10
_CLEANUP_INTERVAL_S = 60
_CHUNK = 1 << 20


class SessionOut(BaseModel):
    """A newly created session."""

    session_id: str
    ttl_minutes: int


class AskIn(BaseModel):
    """A natural-language question."""

    question: str = Field(min_length=1, max_length=2000)


class ConnectIn(BaseModel):
    """External database connection request (disabled unless the server enables it)."""

    url: str = Field(min_length=8, max_length=500)


class PreviewOut(BaseModel):
    """First rows of a table."""

    table: str
    columns: list[str]
    rows: list[list[Any]]


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def describe_llm_error(error_class: str, message: str) -> str:
    """Turn provider failures into messages a user can act on (no internals leaked)."""
    if error_class == "LLMQuotaExceeded":
        return "The daily LLM quota is used up. Try again later; identical questions are cached."
    if error_class == "LLMRateLimitError":
        return "The LLM provider is rate limiting requests. Please retry in a minute."
    if error_class in {"LLMError", "LLMTransientError", "LLMOutputError"}:
        return f"The language model could not complete the request: {message[:200]}"
    return "Something went wrong while answering. Please try again."


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    """Build the app. ``llm`` can be injected (tests); otherwise built lazily from settings."""
    settings = settings or get_settings()
    store = SessionStore(settings)
    holder: dict[str, LLM] = {"llm": llm} if llm else {}

    def get_llm() -> LLM:
        if "llm" not in holder:
            try:
                holder["llm"] = LLM(settings)
            except LLMError as exc:
                raise HTTPException(503, f"LLM is not configured: {exc}") from exc
        return holder["llm"]

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async def sweeper() -> None:
            while True:
                await asyncio.sleep(_CLEANUP_INTERVAL_S)
                await asyncio.to_thread(store.cleanup_expired)

        task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="DataSleuth", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(SessionNotFound)
    async def _not_found(_r: Request, _e: SessionNotFound) -> JSONResponse:
        return JSONResponse({"detail": "Session not found or expired."}, status_code=404)

    @app.exception_handler(UploadError)
    async def _bad_upload(_r: Request, exc: UploadError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def _unexpected(_r: Request, exc: Exception) -> JSONResponse:
        error_id = uuid.uuid4().hex[:8]
        log.exception("unhandled error %s", error_id, exc_info=exc)
        return JSONResponse({"detail": f"Internal error (reference {error_id})."}, status_code=500)

    # ------------------------------------------------------------- sessions
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", response_model=SessionOut, status_code=201)
    def create_session() -> SessionOut:
        session = store.create()
        return SessionOut(session_id=session.id, ttl_minutes=settings.session_ttl_min)

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        store.get(session_id, touch=False)
        store.delete(session_id)

    @app.post("/sessions/{session_id}/upload", response_model=SchemaInfo)
    async def upload(session_id: str, files: list[UploadFile] = File(...)) -> SchemaInfo:  # noqa: B008
        session = store.get(session_id)
        if len(files) > _MAX_FILES_PER_UPLOAD:
            raise HTTPException(400, f"At most {_MAX_FILES_PER_UPLOAD} files per upload.")
        staging = session.directory / "staging"
        staging.mkdir(exist_ok=True)
        saved: list[tuple[Path, str]] = []
        limit = settings.upload_max_mb * 1024 * 1024
        try:
            for i, up in enumerate(files):
                display = Path(up.filename or "upload").name[:120]
                dest = staging / f"{i}.bin"  # never derived from the client filename
                size = 0
                with dest.open("wb") as fh:
                    while chunk := await up.read(_CHUNK):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(
                                413,
                                f"'{display}' exceeds the {settings.upload_max_mb} MB upload limit.",
                            )
                        fh.write(chunk)
                saved.append((dest, display))
            return await asyncio.to_thread(store.add_files, session, saved)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @app.post("/sessions/{session_id}/connect", response_model=SchemaInfo)
    def connect(session_id: str, body: ConnectIn) -> SchemaInfo:
        session = store.get(session_id)
        if not settings.allow_db_connect:
            raise HTTPException(403, "Connecting external databases is disabled on this server.")
        if body.url.lower().startswith("sqlite"):
            raise HTTPException(400, "SQLite URLs are not accepted here; upload the file instead.")
        try:
            return store.connect_database(session, body.url)
        except ConnectionRejected as exc:
            raise HTTPException(400, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except Exception as exc:
            log.warning("database connection failed: %s", type(exc).__name__)
            raise HTTPException(
                400, "Could not connect to the database with those settings."
            ) from exc

    @app.get("/sessions/{session_id}/schema", response_model=SchemaInfo)
    def get_schema(session_id: str) -> SchemaInfo:
        session = store.get(session_id)
        schema = store.schema(session)
        if schema is None:
            raise HTTPException(404, "No data yet. Upload a CSV/XLSX file first.")
        return schema

    @app.get("/sessions/{session_id}/preview/{table}", response_model=PreviewOut)
    def preview(session_id: str, table: str, limit: int = 20) -> PreviewOut:
        session = store.get(session_id)
        schema = store.schema(session)
        info = schema.table(table) if schema else None
        if info is None:
            raise HTTPException(404, "Unknown table.")
        source = store.source(session)
        quote = (
            source.engine.dialect.identifier_preparer.quote
            if source.engine is not None
            else (lambda n: '"' + n.replace('"', '""') + '"')
        )
        n = max(1, min(limit, 100))
        result = Sandbox(source, settings).execute(
            "sql", f"SELECT * FROM {quote(info.name)} LIMIT {n}"
        )
        if not result.ok:
            raise HTTPException(500, "Could not read the table preview.")
        return PreviewOut(table=info.name, columns=result.columns, rows=result.rows)

    # ------------------------------------------------------------------ ask
    @app.post("/sessions/{session_id}/ask")
    def ask(session_id: str, body: AskIn) -> StreamingResponse:
        session = store.get(session_id)
        schema = store.schema(session)
        if schema is None:
            raise HTTPException(409, "Upload data before asking questions.")
        agent =DataSleuthAgent(
            store.source(session),
            settings,
            llm=get_llm(),
            schema=schema,
            checkpoint_path=settings.data_dir / "checkpoints.sqlite",
            artifacts_root=session.traces_dir,
        )
        trace_id = uuid.uuid4().hex

        def events() -> Iterator[str]:
            yield _sse("start", {"trace_id": trace_id})
            for ev in agent.stream(body.question, trace_id=trace_id):
                kind = ev["type"]
                if kind == "result":
                    yield _sse("result", ev["report"].model_dump(mode="json"))
                elif kind == "error":
                    yield _sse(
                        "error", {"message": describe_llm_error(ev["error_class"], ev["message"])}
                    )
                else:
                    yield _sse("progress", ev)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    # --------------------------------------------------------------- traces
    @app.get("/traces/{trace_id}", response_model=Report)
    def get_trace(trace_id: str) -> Report:
        directory = store.find_trace(trace_id)
        if directory is None:
            raise HTTPException(404, "Trace not found.")
        return Report.model_validate_json((directory / "trace.json").read_text(encoding="utf-8"))

    @app.get("/traces/{trace_id}/chart.png")
    def get_chart_png(trace_id: str) -> FileResponse:
        directory = store.find_trace(trace_id)
        png = directory / "chart.png" if directory else None
        if png is None or not png.exists():
            raise HTTPException(404, "No chart for this trace.")
        return FileResponse(png, media_type="image/png")

    if settings.frontend_dist.exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")
    return app
