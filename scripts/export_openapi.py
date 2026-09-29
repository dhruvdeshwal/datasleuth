"""Write the FastAPI OpenAPI schema to docs/openapi.json (input for the TS client generator)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from datasleuth.api.app import create_app
from datasleuth.config import Settings


def main() -> None:
    """Export the schema without touching real data directories."""
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Settings(data_dir=Path(tmp), frontend_dist=Path(tmp) / "none"))
        schema = app.openapi()
    out = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
