"""Ingestion, cleaning and profiling on messy data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from datasleuth.config import Settings
from datasleuth.db.cleaning import apply_plan, clean_headers, infer_plan
from datasleuth.db.ingest import UploadError, ingest_files, validate_upload

MESSY = """Quarterly export - do not edit,,,,
 Order ID ,Customer Name,Order Date,Amount,Discount,Amount,Active,Zip
1001, Alice ,2023-01-15,"$1,200.50",10%,5,yes,02134
1002,Bob,01/20/2023,$300.00,5%,7,no,90210
1003,,2023-02-03,N/A,,9,yes,10001
1004,Dana,03/15/2023,"$2,000",12.5%,1,no,00501
"""


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(llm_provider="fake", data_dir=tmp_path)


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_clean_headers_dedupes_and_snake_cases() -> None:
    names, originals = clean_headers([" Order ID ", "Amount", "Amount", None, "1st"])
    assert names == ["order_id", "amount", "amount_2", "col_4", "c_1st"]
    assert originals[0] == "Order ID"


def test_numeric_currency_percent_and_zip_preserved() -> None:
    money = pd.Series(["$1,200.50", "(300)", "2,000", None])
    plan = infer_plan(money)
    assert plan.dtype == "real" and plan.semantic and "currency" in plan.semantic
    values, bad = apply_plan(money, plan)
    assert list(values.dropna()) == [1200.5, -300.0, 2000.0] and bad == 0
    assert infer_plan(pd.Series(["02134", "90210", "00501"])).dtype == "text"
    pct = infer_plan(pd.Series(["10%", "5%", "12.5%"]))
    assert pct.dtype == "real" and "percent" in (pct.semantic or "")


def test_mixed_date_formats_normalised_to_iso() -> None:
    col = pd.Series(["2023-01-15", "01/20/2023", "2023-02-03", "03/15/2023"])
    plan = infer_plan(col)
    assert plan.dtype == "date"
    out, bad = apply_plan(col, plan)
    assert list(out) == ["2023-01-15", "2023-01-20", "2023-02-03", "2023-03-15"] and bad == 0


def test_mostly_numeric_with_junk_coerces_and_counts() -> None:
    col = pd.Series(["1", "2", "3", "4", "5", "6", "7", "8", "9", "oops"])
    plan = infer_plan(col)
    assert plan.dtype == "integer"
    _, bad = apply_plan(col, plan)
    assert bad == 1


def test_ingest_messy_csv_end_to_end(tmp_path: Path, settings: Settings) -> None:
    path = write(tmp_path, "sales export.csv", MESSY)
    schema = ingest_files([(path, "sales export.csv")], tmp_path / "s.sqlite", settings)
    table = schema.tables[0]
    assert table.name == "sales_export" and table.row_count == 4
    cols = {c.name: c for c in table.columns}
    assert list(cols) == [
        "order_id",
        "customer_name",
        "order_date",
        "amount",
        "discount",
        "amount_2",
        "active",
        "zip",
    ]
    assert cols["order_id"].dtype == "integer" and cols["order_id"].primary_key
    assert cols["order_date"].dtype == "date"
    assert cols["amount"].dtype == "real" and "currency" in (cols["amount"].semantic or "")
    assert cols["discount"].semantic and "percent" in cols["discount"].semantic
    assert cols["active"].dtype == "boolean"
    assert cols["zip"].dtype == "text"
    assert cols["customer_name"].null_count == 1 and cols["customer_name"].samples
    with sqlite3.connect(tmp_path / "s.sqlite") as conn:
        got = conn.execute(
            "SELECT customer_name, amount, order_date, zip FROM sales_export ORDER BY 1"
        ).fetchall()
    assert (None, None, "2023-02-03", "10001") in got
    assert ("Alice", 1200.5, "2023-01-15", "02134") in got


def test_multi_sheet_xlsx_and_fk_inference(tmp_path: Path, settings: Settings) -> None:
    xlsx = tmp_path / "shop.xlsx"
    with pd.ExcelWriter(xlsx) as xw:
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}).to_excel(
            xw, sheet_name="customers", index=False
        )
        pd.DataFrame(
            {"order_id": [10, 11, 12], "customer_id": [1, 2, 2], "total": [5.0, 6.5, 7.0]}
        ).to_excel(xw, sheet_name="orders", index=False)
    schema = ingest_files([(xlsx, "shop.xlsx")], tmp_path / "x.sqlite", settings)
    assert {t.name for t in schema.tables} == {"shop_customers", "shop_orders"}
    orders = schema.table("shop_orders")
    assert orders and orders.row_count == 3
    assert [(fk.columns, fk.ref_table) for fk in orders.foreign_keys] == [
        (["customer_id"], "shop_customers")
    ]


def test_tsv_and_latin1_fallback(tmp_path: Path, settings: Settings) -> None:
    p = tmp_path / "t.tsv"
    p.write_bytes("name\tcity\nJosé\tZürich\nAna\tOslo\n".encode("latin-1"))
    schema = ingest_files([(p, "t.tsv")], tmp_path / "t.sqlite", settings)
    assert schema.tables[0].row_count == 2


@pytest.mark.parametrize(
    ("name", "content", "msg"),
    [
        ("evil.exe", b"MZ", "Unsupported"),
        ("e.csv", b"", "empty"),
        ("b.csv", b"a\x00b\x00" * 10, "binary"),
    ],
)
def test_upload_validation(tmp_path: Path, name: str, content: bytes, msg: str) -> None:
    p = tmp_path / "up"
    p.write_bytes(content)
    with pytest.raises(UploadError, match=msg):
        validate_upload(p, name, 200)


def test_upload_size_limit(tmp_path: Path) -> None:
    p = tmp_path / "big.csv"
    p.write_bytes(b"a,b\n" + b"1,2\n" * 300_000)
    with pytest.raises(UploadError, match="exceeds"):
        validate_upload(p, "big.csv", 1)


def test_large_file_streams_in_chunks(tmp_path: Path, settings: Settings) -> None:
    n = 120_000
    p = tmp_path / "big.csv"
    pd.DataFrame({"k": range(n), "v": [f"${i}.50" for i in range(n)]}).to_csv(p, index=False)
    schema = ingest_files([(p, "big.csv")], tmp_path / "b.sqlite", settings)
    assert schema.tables[0].row_count == n
    assert schema.tables[0].warnings and "first 50000 rows" in schema.tables[0].warnings[0]
