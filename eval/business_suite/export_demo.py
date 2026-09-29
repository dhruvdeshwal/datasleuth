"""Export a compact CSV slice of Northwind for trying the UI (2022-2023 orders and their lines)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

from eval.business_suite.build import DB, download

QUERIES = {
    "customers": "SELECT CustomerID, CompanyName, City, Country FROM Customers",
    "products": "SELECT ProductID, ProductName, CategoryID, SupplierID, UnitPrice, UnitsInStock, ReorderLevel FROM Products",
    "categories": "SELECT CategoryID, CategoryName FROM Categories",
    "orders": "SELECT OrderID, CustomerID, EmployeeID, OrderDate, ShippedDate, Freight, ShipCountry FROM Orders WHERE substr(OrderDate, 1, 4) IN ('2022', '2023')",
    "order_lines": "SELECT od.OrderID, od.ProductID, od.UnitPrice, od.Quantity, od.Discount FROM \"Order Details\" od JOIN Orders o ON od.OrderID = o.OrderID WHERE substr(o.OrderDate, 1, 4) IN ('2022', '2023')",
}


def export(out: Path) -> list[Path]:
    """Write one CSV per table and return the paths."""
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(download(DB))
    try:
        paths = []
        for name, sql in QUERIES.items():
            path = out / f"northwind_{name}.csv"
            pd.read_sql_query(sql, con).to_csv(path, index=False)
            paths.append(path)
        return paths
    finally:
        con.close()


if __name__ == "__main__":
    for p in export(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/northwind")):
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")
