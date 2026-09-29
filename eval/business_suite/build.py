"""Business case study: questions a manager would ask of the Northwind trading database.

Northwind (customers, orders, order lines, products, suppliers, employees) is a classic
business sample. This copy has 16,282 orders and 609,283 order lines, so it doubles as a
scale test. Expected answers are computed with pandas on the raw tables, independently of the
gold SQL; ``tests/test_business_suite.py`` checks that the two agree.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from eval.custom_suite.build import QA, rows

ROOT = Path(__file__).parent
DB = ROOT / "northwind.db"
URL = "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db"
REV = "UnitPrice * Quantity * (1 - Discount)"


def download(path: Path = DB) -> Path:
    """Fetch the Northwind SQLite file (~25 MB, public sample database)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path


def load_frames(path: Path = DB) -> dict[str, pd.DataFrame]:
    """Raw tables as DataFrames (for the independent expected answers)."""
    con = sqlite3.connect(path)
    try:
        names = [
            "Categories",
            "Customers",
            "Employees",
            "Order Details",
            "Orders",
            "Products",
            "Shippers",
            "Suppliers",
        ]
        return {n: pd.read_sql_query(f'SELECT * FROM "{n}"', con) for n in names}
    finally:
        con.close()


def _lines(f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    d = f["Order Details"].copy()
    d["rev"] = d.UnitPrice * d.Quantity * (1 - d.Discount)
    o = f["Orders"][["OrderID", "CustomerID", "EmployeeID", "OrderDate", "ShipVia", "Freight"]]
    return d.merge(o, on="OrderID")


def _year(s: pd.Series) -> pd.Series:
    return s.str[:4]


def questions() -> list[tuple[QA, str]]:
    """(QA, difficulty) pairs. Difficulty is a manual label for reporting."""
    d, e, m, h = "easy", "medium", "medium", "hard"
    L = _lines
    q: list[tuple[QA, str]] = [
        (
            QA(
                "biz-01",
                "How many customers do we have?",
                "SELECT COUNT(*) FROM Customers",
                lambda f: rows(len(f["Customers"])),
            ),
            d,
        ),
        (
            QA(
                "biz-02",
                "How many orders have not been shipped yet?",
                "SELECT COUNT(*) FROM Orders WHERE ShippedDate IS NULL",
                lambda f: rows(f["Orders"].ShippedDate.isna().sum()),
            ),
            d,
        ),
        (
            QA(
                "biz-03",
                "How many products are discontinued?",
                "SELECT COUNT(*) FROM Products WHERE Discontinued = 1",
                lambda f: rows((pd.to_numeric(f["Products"].Discontinued) == 1).sum()),
            ),
            d,
        ),
        (
            QA(
                "biz-04",
                "What is the total value of our current inventory (units in stock times unit price)?",
                "SELECT SUM(UnitsInStock * UnitPrice) FROM Products",
                lambda f: rows((f["Products"].UnitsInStock * f["Products"].UnitPrice).sum()),
            ),
            d,
        ),
        (
            QA(
                "biz-05",
                "Which country has the most customers?",
                "SELECT Country FROM Customers GROUP BY Country ORDER BY COUNT(*) DESC LIMIT 1",
                lambda f: rows(f["Customers"].Country.value_counts().index[0]),
            ),
            d,
        ),
        (
            QA(
                "biz-06",
                "What is our total revenue after discounts?",
                f'SELECT SUM({REV}) FROM "Order Details"',
                lambda f: rows(L(f).rev.sum()),
            ),
            e,
        ),
        (
            QA(
                "biz-07",
                "What is the revenue after discounts for each product category?",
                'SELECT c.CategoryName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID GROUP BY c.CategoryName',
                lambda f: rows(
                    L(f)
                    .merge(f["Products"][["ProductID", "CategoryID"]], on="ProductID")
                    .merge(f["Categories"][["CategoryID", "CategoryName"]], on="CategoryID")
                    .groupby("CategoryName", as_index=False)
                    .rev.sum()
                ),
            ),
            e,
        ),
        (
            QA(
                "biz-08",
                "Who are our top 5 customers by revenue after discounts? Show the company name and the revenue.",
                'SELECT c.CompanyName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS rev FROM "Order Details" od JOIN Orders o ON od.OrderID = o.OrderID JOIN Customers c ON o.CustomerID = c.CustomerID GROUP BY c.CustomerID ORDER BY rev DESC LIMIT 5',
                lambda f: rows(
                    L(f)
                    .groupby("CustomerID", as_index=False)
                    .rev.sum()
                    .sort_values("rev", ascending=False)
                    .head(5)
                    .merge(f["Customers"][["CustomerID", "CompanyName"]], on="CustomerID")[
                        ["CompanyName", "rev"]
                    ]
                ),
                order=True,
            ),
            e,
        ),
        (
            QA(
                "biz-09",
                "What is the average order value (revenue after discounts per order)?",
                f'SELECT SUM({REV}) / COUNT(DISTINCT OrderID) FROM "Order Details"',
                lambda f: rows(L(f).rev.sum() / L(f).OrderID.nunique()),
            ),
            e,
        ),
        (
            QA(
                "biz-10",
                "How many orders were shipped after their required date?",
                "SELECT COUNT(*) FROM Orders WHERE ShippedDate > RequiredDate",
                lambda f: rows((f["Orders"].ShippedDate > f["Orders"].RequiredDate).sum()),
            ),
            e,
        ),
        (
            QA(
                "biz-11",
                "What is the monthly revenue after discounts for 2022?",
                "SELECT substr(o.OrderDate, 1, 7) AS month, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) FROM \"Order Details\" od JOIN Orders o ON od.OrderID = o.OrderID WHERE substr(o.OrderDate, 1, 4) = '2022' GROUP BY month ORDER BY month",
                lambda f: rows(
                    L(f)
                    .loc[lambda x: _year(x.OrderDate) == "2022"]
                    .assign(month=lambda x: x.OrderDate.str[:7])
                    .groupby("month", as_index=False)
                    .rev.sum()
                ),
                order=True,
            ),
            h,
        ),
        (
            QA(
                "biz-12",
                "Which employee generated the most revenue after discounts? Give their full name.",
                "SELECT e.FirstName || ' ' || e.LastName FROM \"Order Details\" od JOIN Orders o ON od.OrderID = o.OrderID JOIN Employees e ON o.EmployeeID = e.EmployeeID GROUP BY e.EmployeeID ORDER BY SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) DESC LIMIT 1",
                lambda f: rows(
                    (lambda t: t.FirstName + " " + t.LastName)(
                        L(f)
                        .groupby("EmployeeID")
                        .rev.sum()
                        .sort_values(ascending=False)
                        .head(1)
                        .reset_index()
                        .merge(f["Employees"], on="EmployeeID")
                    ).iloc[0]
                ),
            ),
            e,
        ),
        (
            QA(
                "biz-13",
                "What is the average freight cost for each shipping company?",
                "SELECT s.CompanyName, AVG(o.Freight) FROM Orders o JOIN Shippers s ON o.ShipVia = s.ShipperID GROUP BY s.CompanyName",
                lambda f: rows(
                    f["Orders"]
                    .merge(f["Shippers"], left_on="ShipVia", right_on="ShipperID")
                    .groupby("CompanyName", as_index=False)
                    .Freight.mean()
                ),
            ),
            e,
        ),
        (
            QA(
                "biz-14",
                "Which products are running low, meaning units in stock are below the reorder level? List their names.",
                "SELECT ProductName FROM Products WHERE UnitsInStock < ReorderLevel",
                lambda f: rows(
                    f["Products"].loc[lambda x: x.UnitsInStock < x.ReorderLevel, ["ProductName"]]
                ),
            ),
            e,
        ),
        (
            QA(
                "biz-15",
                "Which suppliers provide more than 3 products? List the company names.",
                "SELECT s.CompanyName FROM Suppliers s JOIN Products p ON s.SupplierID = p.SupplierID GROUP BY s.SupplierID HAVING COUNT(*) > 3",
                lambda f: rows(
                    f["Products"]
                    .merge(f["Suppliers"], on="SupplierID")
                    .groupby(["SupplierID", "CompanyName"])
                    .size()
                    .loc[lambda s: s > 3]
                    .reset_index()[["CompanyName"]]
                ),
            ),
            h,
        ),
        (
            QA(
                "biz-16",
                "By what percentage did revenue after discounts grow from 2021 to 2022?",
                "SELECT 100.0 * (SUM(CASE WHEN substr(o.OrderDate, 1, 4) = '2022' THEN od.UnitPrice * od.Quantity * (1 - od.Discount) END) / SUM(CASE WHEN substr(o.OrderDate, 1, 4) = '2021' THEN od.UnitPrice * od.Quantity * (1 - od.Discount) END) - 1) FROM \"Order Details\" od JOIN Orders o ON od.OrderID = o.OrderID",
                lambda f: rows(
                    100.0
                    * (
                        L(f).loc[lambda x: _year(x.OrderDate) == "2022"].rev.sum()
                        / L(f).loc[lambda x: _year(x.OrderDate) == "2021"].rev.sum()
                        - 1
                    )
                ),
            ),
            h,
        ),
        (
            QA(
                "biz-17",
                "What share of total revenue after discounts comes from our top 10 customers? Give the percentage.",
                f'SELECT 100.0 * SUM(rev) / (SELECT SUM({REV}) FROM "Order Details") FROM (SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS rev FROM "Order Details" od JOIN Orders o ON od.OrderID = o.OrderID GROUP BY o.CustomerID ORDER BY rev DESC LIMIT 10)',
                lambda f: rows(
                    100.0
                    * L(f)
                    .groupby("CustomerID")
                    .rev.sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .sum()
                    / L(f).rev.sum()
                ),
            ),
            h,
        ),
        (
            QA(
                "biz-18",
                "What was our best-selling product by quantity in 2023?",
                "SELECT p.ProductName FROM \"Order Details\" od JOIN Orders o ON od.OrderID = o.OrderID JOIN Products p ON od.ProductID = p.ProductID WHERE substr(o.OrderDate, 1, 4) = '2023' GROUP BY p.ProductID ORDER BY SUM(od.Quantity) DESC LIMIT 1",
                lambda f: rows(
                    L(f)
                    .loc[lambda x: _year(x.OrderDate) == "2023"]
                    .groupby("ProductID")
                    .Quantity.sum()
                    .sort_values(ascending=False)
                    .head(1)
                    .reset_index()
                    .merge(f["Products"], on="ProductID")
                    .ProductName.iloc[0]
                ),
            ),
            h,
        ),
        (
            QA(
                "biz-19",
                "How many orders did each employee handle in 2023? Show the employee's last name and the count.",
                "SELECT e.LastName, COUNT(*) FROM Orders o JOIN Employees e ON o.EmployeeID = e.EmployeeID WHERE substr(o.OrderDate, 1, 4) = '2023' GROUP BY e.EmployeeID",
                lambda f: rows(
                    f["Orders"]
                    .loc[lambda x: _year(x.OrderDate) == "2023"]
                    .merge(f["Employees"], on="EmployeeID")
                    .groupby(["EmployeeID", "LastName"])
                    .size()
                    .reset_index()[["LastName", 0]]
                ),
            ),
            m,
        ),
        (
            QA(
                "biz-20",
                "How many customers have placed more than 170 orders?",
                "SELECT COUNT(*) FROM (SELECT CustomerID FROM Orders GROUP BY CustomerID HAVING COUNT(*) > 170)",
                lambda f: rows((f["Orders"].groupby("CustomerID").size() > 170).sum()),
            ),
            e,
        ),
        (
            QA(
                "biz-21",
                "What is the average discount given on an order line?",
                'SELECT AVG(Discount) FROM "Order Details"',
                lambda f: rows(f["Order Details"].Discount.mean()),
            ),
            d,
        ),
        (QA("biz-22", "What is our profit margin for each product?", "", None), d),
        (QA("biz-23", "What is the customer satisfaction score for each product?", "", None), d),
    ]
    return q


def build(root: Path = ROOT, frames: dict[str, pd.DataFrame] | None = None) -> list[dict[str, Any]]:
    """Compute expected answers with pandas and write ``qa.json``."""
    f = frames or load_frames(root / "northwind.db")
    out = [
        {
            "qid": qa.qid, "dataset": "northwind", "question": qa.question, "gold_sql": qa.gold_sql,
            "expected": None if qa.expected is None else qa.expected(f), "order_matters": qa.order,
            "difficulty": diff if qa.expected is not None else "unanswerable", "answerable": qa.expected is not None,
        }
        for qa, diff in questions()
    ]  # fmt: skip
    (root / "qa.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def load_cases(root: Path = ROOT) -> list[Any]:
    """Evaluation cases against the Northwind database (downloading it on first use)."""
    from eval.systems import Case

    download(root / "northwind.db")
    if not (root / "qa.json").exists():
        build(root)
    return [
        Case(
            qid=i["qid"], db_path=root / "northwind.db", question=i["question"], gold_sql=i["gold_sql"],
            difficulty=i["difficulty"], group="northwind", lenient=True,
        )
        for i in json.loads((root / "qa.json").read_text(encoding="utf-8"))
    ]  # fmt: skip


if __name__ == "__main__":
    download()
    print(f"{len(build())} questions written")
