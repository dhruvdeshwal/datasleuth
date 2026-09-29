# Error analysis: business_full__gemini-3.1-flash-lite__n23_s0

2 failures out of 20 questions.

| Cluster | Count | Meaning |
|---|---:|---|
| wrong_grouping | 1 | Grouped by a different column (e.g. a non-unique name instead of the key), merging or splitting entities |
| value_format_mismatch | 1 | Filter literals differ from the gold ones (case, spelling, format), often giving empty/fewer rows |

## wrong_grouping (1)

Grouped by a different column (e.g. a non-unique name instead of the key), merging or splitting entities

**biz-08** (medium) — Who are our top 5 customers by revenue after discounts? Show the company name and the revenue.

- Gold: `SELECT c.CompanyName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS rev FROM "Order Details" od JOIN Orders o ON od.OrderID = o.OrderID JOIN Customers c ON o.CustomerID = c.CustomerID GROUP BY c.CustomerID ORDER BY rev DESC LIMIT 5`
- Predicted: `SELECT T1.CompanyName, SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) AS Revenue FROM Customers AS T1 JOIN Orders AS T3 ON T1.CustomerID = T3.CustomerID JOIN "Order Details" AS T2 ON T3.OrderID = T2.OrderID GROUP BY T1.CompanyName ORDER BY Revenue DESC LIMIT 5`
- Predicted rows (first 3): `[["IT", 9745371.29], ["B's Beverages", 6154115.34], ["Hungry Coyote Import Store", 5698023.67]]`
- Note: A real data trap: two different customers (IDs Val2 and VALON) share the company name "IT". Grouping by name merges them into a fake 9.7M "top customer". The gold groups by customer ID.

## value_format_mismatch (1)

Filter literals differ from the gold ones (case, spelling, format), often giving empty/fewer rows

**biz-11** (hard) — What is the monthly revenue after discounts for 2022?

- Gold: `SELECT substr(o.OrderDate, 1, 7) AS month, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) FROM "Order Details" od JOIN Orders o ON od.OrderID = o.OrderID WHERE substr(o.OrderDate, 1, 4) = '2022' GROUP BY month ORDER BY month`
- Predicted: `SELECT strftime('%m', "OrderDate") AS "Month", SUM("UnitPrice" * "Quantity" * (1 - "Discount")) AS "MonthlyRevenue" FROM "Orders" JOIN "Order Details" ON "Orders"."OrderID" = "Order Details"."OrderID" WHERE "OrderDate" BETWEEN '2022-01-01' AND '2022-12-31 23:59:59' GROUP BY "Month"`
- Predicted rows (first 3): `[["01", 3591353.72], ["02", 2832321.16], ["03", 3222005.69]]`
- Note: The numbers match the gold answer exactly; the model labelled months '01'..'12' (the year was filtered to 2022 in the query) while the gold uses 'YYYY-MM'. Arguably a correct answer scored as wrong by exact comparison. The question was NOT edited after seeing the output.

## More failing examples

