"""Header cleaning and column type inference for messy tabular data.

Type inference runs on a sample and yields a :class:`ColumnPlan`, which is then
applied chunk-by-chunk so arbitrarily large files never have to fit in memory.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "#n/a", "?", "missing"}
_TRUE = {"true", "yes", "y", "t"}
_FALSE = {"false", "no", "n", "f"}
_CURRENCY_CHARS = "$€£¥₹"
_NUMBER_RE = r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?$|^[-+]?\.\d+$"
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%d/%m/%y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%Y-%b-%d",
]
_MIN_COVERAGE = 0.9


@dataclass
class ColumnPlan:
    """How to convert one raw text column."""

    dtype: str = "text"
    date_formats: list[str] = field(default_factory=list)
    semantic: str | None = None
    notes: list[str] = field(default_factory=list)


def clean_headers(raw: list[object]) -> tuple[list[str], list[str]]:
    """Return (snake_case unique names, original labels) for raw header cells."""
    names: list[str] = []
    originals: list[str] = []
    seen: dict[str, int] = {}
    for i, cell in enumerate(raw, start=1):
        original = "" if cell is None or (isinstance(cell, float) and pd.isna(cell)) else str(cell)
        text = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
        text = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip()).strip("_").lower()
        if not text:
            text = f"col_{i}"
        if text[0].isdigit():
            text = f"c_{text}"
        base, n = text, seen.get(text, 0)
        seen[base] = n + 1
        if n:
            text = f"{base}_{n + 1}"
            while text in seen:
                n += 1
                text = f"{base}_{n + 1}"
            seen[text] = 1
        names.append(text)
        originals.append(original.strip())
    return names, originals


def clean_table_name(stem: str, taken: set[str]) -> str:
    """Make a safe, unique SQL table name from a file/sheet name."""
    names, _ = clean_headers([stem])
    base = names[0] if names[0] != "col_1" else "table"
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}_{n}"
    return name


def detect_header_row(rows: pd.DataFrame, max_scan: int = 10) -> int:
    """Index of the header row, skipping leading title/blank rows.

    A leading row is treated as a title when it has far fewer filled cells than
    the rows that follow it.
    """
    filled = rows.notna().sum(axis=1).tolist()
    if not filled:
        return 0
    width = max(filled[: max_scan + 5] or [0])
    for i, count in enumerate(filled[:max_scan]):
        if count >= max(1, 0.6 * width):
            return i
    return 0


def normalize_cells(series: pd.Series) -> pd.Series:
    """Strip whitespace, map null-like tokens to None and render dates as ISO text."""

    def fix(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, datetime | pd.Timestamp):
            return value.strftime(
                "%Y-%m-%d %H:%M:%S" if (value.hour or value.minute or value.second) else "%Y-%m-%d"
            )
        if isinstance(value, date):
            return value.isoformat()
        text = str(value).strip()
        return None if text.lower() in NULL_TOKENS else text

    return series.map(fix).astype("object")


def _numeric(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parse strings like ``$1,234.50``, ``(12)`` and ``15%`` → (values, has_currency, has_pct)."""
    t = s.astype("string")
    neg = t.str.match(r"^\(.*\)$").fillna(False).astype(bool)
    has_cur = t.str.contains(f"[{_CURRENCY_CHARS}]").fillna(False).astype(bool)
    has_pct = t.str.endswith("%").fillna(False).astype(bool)
    core = t.str.replace(rf"[()%\s{_CURRENCY_CHARS}]", "", regex=True)
    valid = core.str.match(_NUMBER_RE).fillna(False).astype(bool)
    nums = pd.to_numeric(core.str.replace(",", "", regex=False).where(valid), errors="coerce")
    nums = nums.where(~neg, -nums.abs()).astype("float64")
    return nums, has_cur, has_pct


def _parse_dates(s: pd.Series, formats: list[str]) -> pd.Series:
    """Parse with each format in turn; later formats only fill still-unparsed cells."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in formats:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        parsed = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        out.loc[todo] = parsed.astype("datetime64[ns]")
    return out


def infer_plan(series: pd.Series) -> ColumnPlan:
    """Infer the best type for a normalised text column from a sample."""
    s = normalize_cells(series)
    values = s.dropna()
    plan = ColumnPlan()
    if values.empty:
        return plan
    lowered = values.str.lower()
    if lowered.isin(_TRUE | _FALSE).all():
        return ColumnPlan(dtype="boolean", semantic="boolean (1 = true, 0 = false)")

    nums, has_cur, has_pct = _numeric(values)
    leading_zero = (
        values.str.match(r"^[-+]?0\d+$").any()
        or values.str.replace(r"\D", "", regex=True).str.len().max() > 15
    )
    if not leading_zero and nums.notna().mean() >= _MIN_COVERAGE:
        ok = nums.dropna()
        integral = bool((ok == ok.round()).all()) and not has_pct[nums.notna()].any()
        plan.dtype = "integer" if integral and ok.abs().max() < 2**62 else "real"
        if has_cur[nums.notna()].mean() >= 0.2:
            plan.semantic = "currency (symbols stripped)"
        elif has_pct[nums.notna()].mean() >= 0.2:
            plan.semantic = "percent (12 means 12%)"
        return plan

    coverage: list[tuple[float, int, str]] = []
    for order, fmt in enumerate(_DATE_FORMATS):
        ratio = pd.to_datetime(values, format=fmt, errors="coerce").notna().mean()
        if ratio > 0:
            coverage.append((float(ratio), -order, fmt))
    if coverage:
        coverage.sort(reverse=True)
        chosen: list[str] = []
        parsed_total = pd.Series(False, index=values.index)
        for _, _, fmt in coverage:
            ok = pd.to_datetime(values, format=fmt, errors="coerce").notna()
            if (ok & ~parsed_total).any():
                chosen.append(fmt)
                parsed_total |= ok
            if parsed_total.mean() >= _MIN_COVERAGE and len(chosen) >= 1 and parsed_total.all():
                break
        if parsed_total.mean() >= _MIN_COVERAGE:
            plan.dtype = "date"
            plan.date_formats = chosen
            if len(chosen) > 1:
                plan.notes.append("mixed date formats were normalised to ISO (YYYY-MM-DD)")
            slash = [f for f in chosen if f in ("%m/%d/%Y", "%d/%m/%Y")]
            if slash:
                reading = "month/day/year" if slash[0] == "%m/%d/%Y" else "day/month/year"
                plan.notes.append(
                    f"slash-separated dates were read as {reading} (chosen from the data)"
                )
            return plan
    return plan


def apply_plan(series: pd.Series, plan: ColumnPlan) -> tuple[pd.Series, int]:
    """Convert a raw column per ``plan``; returns (converted, count of values set to NULL)."""
    s = normalize_cells(series)
    present = s.notna()
    if plan.dtype == "text":
        return s, 0
    if plan.dtype == "boolean":
        low = s.str.lower()
        out = pd.Series(pd.NA, index=s.index, dtype="Int64")
        out[low.isin(_TRUE)] = 1
        out[low.isin(_FALSE)] = 0
        return out, int((present & out.isna()).sum())
    if plan.dtype in ("integer", "real"):
        nums, _, _ = _numeric(s)
        bad = int((present & nums.isna()).sum())
        if plan.dtype == "integer":
            return nums.round().astype("Int64"), bad
        return nums, bad
    parsed = _parse_dates(s, plan.date_formats)
    bad = int((present & parsed.isna()).sum())
    has_time = (
        parsed.dt.hour.fillna(0) + parsed.dt.minute.fillna(0) + parsed.dt.second.fillna(0)
    ) > 0
    fmt = "%Y-%m-%d %H:%M:%S" if bool(has_time.any()) else "%Y-%m-%d"
    return parsed.dt.strftime(fmt).astype("object").where(parsed.notna(), None), bad
