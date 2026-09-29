"""Load CSV/TSV/XLSX files into an isolated per-session SQLite database."""

from __future__ import annotations

import codecs
import logging
import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

from datasleuth.config import Settings
from datasleuth.db.cleaning import (
    ColumnPlan,
    apply_plan,
    clean_headers,
    clean_table_name,
    detect_header_row,
    infer_plan,
)
from datasleuth.db.connectors import sqlite_engine
from datasleuth.db.profile import profile_schema
from datasleuth.state import ColumnInfo, SchemaInfo, TableInfo

log = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}
_CHUNK_ROWS = 50_000
_SQL_TYPES = {
    "integer": "INTEGER",
    "real": "REAL",
    "text": "TEXT",
    "date": "DATE",
    "boolean": "INTEGER",
}


class UploadError(ValueError):
    """The uploaded file is invalid; the message is safe to show to users."""


def validate_upload(path: Path, display_name: str, max_mb: int) -> None:
    """Check extension, size and basic content type of an uploaded file."""
    ext = Path(display_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '{ext or 'none'}'. Upload CSV, TSV or XLSX.")
    size = path.stat().st_size
    if size == 0:
        raise UploadError(f"'{display_name}' is empty.")
    if size > max_mb * 1024 * 1024:
        raise UploadError(f"'{display_name}' exceeds the {max_mb} MB upload limit.")
    with path.open("rb") as fh:
        head = fh.read(8192)
    if ext == ".xlsx":
        if not zipfile.is_zipfile(path):
            raise UploadError(f"'{display_name}' is not a valid XLSX file.")
    elif b"\x00" in head and not head.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UploadError(f"'{display_name}' looks like a binary file, not text CSV.")


# ------------------------------------------------------------------ raw readers
def _is_utf8(path: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as fh:
            while block := fh.read(1 << 20):
                decoder.decode(block)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _csv_chunks(path: Path, delimiter: str | None) -> Iterator[pd.DataFrame]:
    """Yield header-less all-text chunks. DuckDB sniffs dialect; pandas is the fallback."""
    try:
        if not _is_utf8(path):  # DuckDB would silently drop undecodable rows
            raise ValueError("not UTF-8")
        con = duckdb.connect(":memory:")
        opts = "header=false, all_varchar=true, ignore_errors=true, null_padding=true, sample_size=20000"
        if delimiter:
            opts += f", delim='{delimiter}'"
        cur = con.execute(f"SELECT * FROM read_csv('{path.as_posix()}', {opts})")
        first = True
        while batch := cur.fetchmany(_CHUNK_ROWS):
            df = pd.DataFrame(batch, dtype="object")
            if first and df.shape[1] == 1 and delimiter is None:
                raise ValueError("single column; try another parser")
            first = False
            yield df
        return
    except (duckdb.Error, ValueError) as exc:
        log.info("DuckDB could not read %s (%s); falling back to pandas", path.name, exc)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            yield from pd.read_csv(
                path,
                header=None,
                dtype=str,
                sep=delimiter,
                engine="python",
                encoding=enc,
                on_bad_lines="skip",
                chunksize=_CHUNK_ROWS,
                skip_blank_lines=True,
                keep_default_na=False,
                na_values=[],
            )
            return
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as exc:
            raise UploadError(f"Could not parse CSV: {exc}") from exc
    raise UploadError("Could not decode the file as UTF-8 or Latin-1.")


def _xlsx_sheets(path: Path) -> list[str]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def _xlsx_chunks(path: Path, sheet: str) -> Iterator[pd.DataFrame]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        buf: list[tuple[object, ...]] = []
        width = 0
        for row in wb[sheet].iter_rows(values_only=True):
            width = max(width, len(row))
            buf.append(row)
            if len(buf) >= _CHUNK_ROWS:
                yield pd.DataFrame([r + (None,) * (width - len(r)) for r in buf], dtype="object")
                buf = []
        if buf:
            yield pd.DataFrame([r + (None,) * (width - len(r)) for r in buf], dtype="object")
    finally:
        wb.close()


# ------------------------------------------------------------------- ingestion
def _ingest_table(
    conn: sqlite3.Connection,
    chunks: Iterator[pd.DataFrame],
    table: str,
    source: str,
    sample_rows: int,
) -> TableInfo:
    head: list[pd.DataFrame] = []
    total = 0
    for chunk in chunks:
        head.append(chunk)
        total += len(chunk)
        if total >= sample_rows:
            break
    if not head:
        raise UploadError(f"'{source}' contains no data.")
    sample = pd.concat(head, ignore_index=True)
    sample = sample.dropna(how="all")
    sample = sample.loc[:, sample.notna().any(axis=0)]
    keep_cols = list(sample.columns)
    if sample.empty or not keep_cols:
        raise UploadError(f"'{source}' contains no data.")
    header_idx = detect_header_row(sample)
    names, originals = clean_headers(list(sample.iloc[header_idx]))
    body = sample.iloc[header_idx + 1 :]
    if body.empty:
        raise UploadError(f"'{source}' has a header but no data rows.")

    def shape(df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, keep_cols].dropna(how="all")
        df.columns = names
        return df

    body = shape(body)
    plans: dict[str, ColumnPlan] = {n: infer_plan(body[n]) for n in names}
    ddl = ", ".join(f'"{n}" {_SQL_TYPES[plans[n].dtype]}' for n in names)
    conn.execute(f'CREATE TABLE "{table}" ({ddl})')
    coerced = dict.fromkeys(names, 0)
    rows_written = 0

    def write(df: pd.DataFrame) -> None:
        nonlocal rows_written
        out = {}
        for n in names:
            out[n], bad = apply_plan(df[n], plans[n])
            coerced[n] += bad
        frame = pd.DataFrame(out).astype("object")
        frame = frame.where(frame.notna(), None)
        placeholders = ", ".join("?" * len(names))
        conn.executemany(
            f'INSERT INTO "{table}" VALUES ({placeholders})',
            frame.itertuples(index=False, name=None),
        )
        rows_written += len(df)

    write(body)
    for chunk in chunks:
        write(shape(chunk))
    conn.commit()

    warnings = [
        f"column '{n}': {c} value(s) could not be parsed as {plans[n].dtype} and were set to NULL"
        for n, c in coerced.items()
        if c
    ]
    warnings += [f"column '{n}': {note}" for n in names for note in plans[n].notes]
    columns = [
        ColumnInfo(
            name=n, original_name=o or None, dtype=plans[n].dtype, semantic=plans[n].semantic
        )
        for n, o in zip(names, originals, strict=True)
    ]
    return TableInfo(
        name=table, source=source, row_count=rows_written, columns=columns, warnings=warnings
    )


def ingest_files(files: list[tuple[Path, str]], db_path: Path, settings: Settings) -> SchemaInfo:
    """Ingest ``(path, display_name)`` pairs into ``db_path`` and return the profiled schema.

    ``db_path`` is created fresh; existing tables are kept so files can be added
    incrementally to a session.
    """
    for path, name in files:
        validate_upload(path, name, settings.upload_max_mb)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        taken = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        made: dict[str, TableInfo] = {}
        for path, display in files:
            ext = Path(display).suffix.lower()
            stem = Path(display).stem
            if ext == ".xlsx":
                sheets = _xlsx_sheets(path)
                for sheet in sheets:
                    label = stem if len(sheets) == 1 else f"{stem}_{sheet}"
                    table = clean_table_name(label, taken)
                    try:
                        info = _ingest_table(
                            conn,
                            _xlsx_chunks(path, sheet),
                            table,
                            f"{display} [{sheet}]",
                            settings.profile_sample_rows,
                        )
                    except UploadError:
                        if len(sheets) == 1:
                            raise
                        continue
                    taken.add(table)
                    made[table] = info
            else:
                table = clean_table_name(stem, taken)
                delim = "\t" if ext == ".tsv" else None
                info = _ingest_table(
                    conn, _csv_chunks(path, delim), table, display, settings.profile_sample_rows
                )
                taken.add(table)
                made[table] = info
    finally:
        conn.close()
    engine = sqlite_engine(db_path)
    try:
        return profile_schema(
            engine,
            "sqlite",
            sample_rows=settings.profile_sample_rows,
            overrides=made,
            infer_keys=True,
        )
    finally:
        engine.dispose()
