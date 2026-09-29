
PANDAS RULES (only when language is "pandas")
- Every table is already loaded as a DataFrame variable named after the table (also available as dfs["table"]). `pd`, `np` and `math` are pre-imported.
- Allowed imports: pandas, numpy, math. No file access, no eval/exec, no dunder attributes, no .query()/.eval().
- Dates are ISO strings: convert with pd.to_datetime(...) before date arithmetic.
- Assign the final answer to a variable named `result` (a DataFrame, Series or scalar). Keep it small: aggregate instead of returning raw rows.
- Prefer SQL unless the question needs reshaping (pivot), rolling/window statistics or other things awkward in SQL.
