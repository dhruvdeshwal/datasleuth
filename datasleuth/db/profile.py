"""Schema extraction and data profiling over any SQLAlchemy engine."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from sqlalchemy import Engine, inspect, text
from sqlalchemy import types as sqltypes

from datasleuth.state import ColumnInfo, Dialect, ForeignKey, SchemaInfo, TableInfo

log = logging.getLogger(__name__)
_SAMPLES = 3


def _dtype(col_type: object) -> str:
    if isinstance(col_type, sqltypes.Boolean):
        return "boolean"
    if isinstance(col_type, sqltypes.Integer):
        return "integer"
    if isinstance(col_type, sqltypes.Numeric | sqltypes.Float):
        return "real"
    if isinstance(col_type, sqltypes.Date | sqltypes.DateTime):
        return "date"
    name = str(col_type).upper()
    if any(k in name for k in ("INT", "SERIAL")):
        return "integer"
    if any(k in name for k in ("REAL", "FLOA", "DOUB", "DEC", "NUM")):
        return "real"
    if "BOOL" in name:
        return "boolean"
    if "DATE" in name or "TIME" in name:
        return "date"
    return "text"


def profile_schema(
    engine: Engine,
    dialect: Dialect,
    *,
    sample_rows: int = 50_000,
    overrides: Mapping[str, TableInfo] | None = None,
    infer_keys: bool = False,
) -> SchemaInfo:
    """Reflect and profile every table/view.

    ``overrides`` supplies semantic info (original names, inferred dtype/semantic,
    warnings) known from ingestion; profiling statistics are always recomputed.
    ``infer_keys`` guesses PK/FK from names for uploaded files without constraints.
    """
    insp = inspect(engine)
    q = engine.dialect.identifier_preparer.quote
    tables: list[TableInfo] = []
    names = insp.get_table_names() + insp.get_view_names()
    for tname in sorted(n for n in names if not n.startswith("sqlite_")):
        known = (overrides or {}).get(tname)
        pk_cols = set(insp.get_pk_constraint(tname).get("constrained_columns") or [])
        cols: list[ColumnInfo] = []
        for c in insp.get_columns(tname):
            prior = known.column(c["name"]) if known else None
            cols.append(
                ColumnInfo(
                    name=c["name"],
                    original_name=prior.original_name if prior else None,
                    dtype=prior.dtype if prior else _dtype(c["type"]),
                    semantic=prior.semantic if prior else None,
                    primary_key=c["name"] in pk_cols,
                )
            )
        fks = [
            ForeignKey(
                columns=list(fk["constrained_columns"]),
                ref_table=fk["referred_table"],
                ref_columns=list(fk["referred_columns"]),
            )
            for fk in insp.get_foreign_keys(tname)
            if fk.get("referred_table")
        ]
        info = TableInfo(
            name=tname,
            source=known.source if known else None,
            columns=cols,
            foreign_keys=fks,
            warnings=list(known.warnings) if known else [],
        )
        _fill_stats(engine, info, q(tname), q, sample_rows)
        tables.append(info)
    schema = SchemaInfo(dialect=dialect, tables=tables)
    if infer_keys:
        _infer_keys(engine, schema, sample_rows)
    return schema


def _fill_stats(engine: Engine, info: TableInfo, qtable: str, q, sample_rows: int) -> None:  # type: ignore[no-untyped-def]
    with engine.connect() as conn:
        counts = ", ".join(f"COUNT({q(c.name)})" for c in info.columns)
        row = conn.execute(
            text(f"SELECT COUNT(*){', ' + counts if counts else ''} FROM {qtable}")
        ).one()
        info.row_count = int(row[0])
        for i, col in enumerate(info.columns, start=1):
            col.null_count = info.row_count - int(row[i])
    sampled = info.row_count > sample_rows
    for col in info.columns:
        qc = q(col.name)
        try:
            with engine.connect() as conn:
                lo, hi = conn.execute(text(f"SELECT MIN({qc}), MAX({qc}) FROM {qtable}")).one()
                col.min = None if lo is None else str(lo)
                col.max = None if hi is None else str(hi)
                inner = f"(SELECT {qc} AS v FROM {qtable} LIMIT {sample_rows}) s"
                col.distinct_count = int(
                    conn.execute(text(f"SELECT COUNT(DISTINCT v) FROM {inner}")).scalar_one()
                )
                top = conn.execute(
                    text(
                        f"SELECT v, COUNT(*) AS n FROM {inner} WHERE v IS NOT NULL "
                        f"GROUP BY v ORDER BY n DESC, v LIMIT {_SAMPLES}"
                    )
                ).all()
                col.samples = [str(r[0]) for r in top]
        except Exception:
            log.debug("could not profile %s.%s", info.name, col.name, exc_info=True)
    if sampled:
        info.warnings.append(
            f"distinct counts and sample values are based on the first {sample_rows} rows"
        )


def _singular(name: str) -> str:
    return re.sub(r"(ies)$", "y", name) if name.endswith("ies") else name.rstrip("s")


def _infer_keys(engine: Engine, schema: SchemaInfo, sample_rows: int) -> None:
    """Guess PK/FK relationships for constraint-less uploads (heuristic, name-based)."""
    pks: dict[str, str] = {}
    for tbl in schema.tables:
        if tbl.row_count == 0 or tbl.row_count > sample_rows:
            continue
        for col in tbl.columns:
            looks_like_id = col.name == "id" or col.name.endswith("_id") or col.name.endswith("id")
            if (
                looks_like_id
                and col.null_count == 0
                and col.distinct_count == tbl.row_count
                and not any(c.primary_key for c in tbl.columns)
            ):
                col.primary_key = True
                pks[tbl.name] = col.name
    q = engine.dialect.identifier_preparer.quote
    for tbl in schema.tables:
        for col in tbl.columns:
            if col.primary_key:
                continue
            for other, pk in pks.items():
                if other == tbl.name:
                    continue
                stem = col.name.removesuffix(f"_{pk}") if col.name != pk else ""
                singular = _singular(other)
                named_alike = col.name == pk or (
                    bool(stem) and (singular == stem or singular.endswith(f"_{stem}"))
                )
                if named_alike and _mostly_contained(engine, tbl.name, col.name, other, pk, q):
                    tbl.foreign_keys.append(
                        ForeignKey(columns=[col.name], ref_table=other, ref_columns=[pk])
                    )
                    break


def _mostly_contained(engine: Engine, t: str, c: str, rt: str, rc: str, q) -> bool:  # type: ignore[no-untyped-def]
    sql = (
        f"SELECT COUNT(*), SUM(CASE WHEN {q(c)} IN (SELECT {q(rc)} FROM {q(rt)}) THEN 1 ELSE 0 END) "
        f"FROM {q(t)} WHERE {q(c)} IS NOT NULL"
    )
    with engine.connect() as conn:
        total, hit = conn.execute(text(sql)).one()
    return bool(total) and (hit or 0) / total >= 0.95
