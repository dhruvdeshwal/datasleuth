"""Download Spider 1.0, the official evaluator and (optionally) the test-suite databases.

Cross-platform; ``scripts/download_spider.sh`` is a thin wrapper. Everything lands under
``eval/spider/`` (git-ignored). Existing archives are reused, so an interrupted download can
simply be re-run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent / "spider"
SPIDER_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"  # Spider 1.0 (yale-lily.github.io/spider)
TESTSUITE_ID = "1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w"  # taoyds/test-suite-sql-eval databases (~1.3 GB)
OFFICIAL_REPO = "https://github.com/taoyds/test-suite-sql-eval.git"


def _gdown(file_id: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"reusing {target.name}")
        return
    import gdown

    print(f"downloading {target.name} from Google Drive ...")
    if not gdown.download(id=file_id, output=str(target), quiet=False):
        raise SystemExit(
            f"Could not download {target.name}. Download it manually from "
            f"https://drive.google.com/uc?id={file_id} and place it at {target}."
        )


def fetch_official(root: Path) -> None:
    """Clone the official test-suite evaluator (also provides Spider's difficulty labels)."""
    dest = root / "test-suite-sql-eval"
    if dest.exists():
        return
    subprocess.run(["git", "clone", "--depth", "1", OFFICIAL_REPO, str(dest)], check=True)


def fetch_spider(root: Path) -> None:
    """Download and extract the Spider dataset."""
    if (root / "spider_data" / "dev.json").exists():
        return
    archive = root / "spider.zip"
    _gdown(SPIDER_ID, archive)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(root, members=members)
    archive.unlink()  # the extracted data is what we need; reclaim ~200 MB


def plan_testsuite(
    names: list[tuple[str, int]], dev_dbs: set[str], variants: int, max_file_mb: int
) -> list[str]:
    """Choose which archive members to extract: per dev database the original plus up to
    ``variants`` variants, skipping variants larger than ``max_file_mb`` (some are ~100 MB)."""
    per_db: dict[str, list[tuple[str, int]]] = {}
    for name, size in names:
        parts = name.split("/")
        if (
            len(parts) == 3
            and parts[0] == "database"
            and parts[1] in dev_dbs
            and parts[2].endswith(".sqlite")
        ):
            per_db.setdefault(parts[1], []).append((name, size))
    chosen: list[str] = []
    for db_id, files in per_db.items():
        files.sort(key=lambda f: (f[0].split("/")[2] != f"{db_id}.sqlite", f[0]))  # original first
        kept = [files[0], *[f for f in files[1:] if f[1] <= max_file_mb * 1024 * 1024][:variants]]
        chosen += [n for n, _ in kept]
    return chosen


def fetch_testsuite(root: Path, variants: int = 10, max_file_mb: int = 100) -> None:
    """Download the test-suite database variants and extract a bounded subset for dev databases."""
    archive = root / "testsuite.zip"
    _gdown(TESTSUITE_ID, archive)
    dev_dbs = {
        d["db_id"]
        for d in json.loads((root / "spider_data" / "dev.json").read_text(encoding="utf-8"))
    }
    out = root / "testsuite_databases"
    with zipfile.ZipFile(archive) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
        chosen = plan_testsuite(list(sizes.items()), dev_dbs, variants, max_file_mb)
        need = sum(sizes[n] for n in chosen)
        free = shutil.disk_usage(root).free
        print(f"extracting {len(chosen)} files ({need / 1e6:.0f} MB); {free / 1e9:.1f} GB free")
        if free < need + 1_000_000_000:
            raise SystemExit(
                "Not enough free disk space (keeping 1 GB spare). Lower --variants / --max-file-mb "
                f"or free up space. Needed {need / 1e9:.2f} GB."
            )
        for name in chosen:
            _, db_id, fname = name.split("/")
            target = out / db_id / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with z.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(
        f"test-suite databases in {out}. You can delete {archive.name} to reclaim {archive.stat().st_size / 1e9:.1f} GB."
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--testsuite", action="store_true", help="also fetch the ~1.3 GB test-suite databases"
    )
    ap.add_argument(
        "--variants", type=int, default=10, help="test-suite variants to keep per database"
    )
    ap.add_argument("--max-file-mb", type=int, default=100, help="skip variants larger than this")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    fetch_official(args.root)
    fetch_spider(args.root)
    if args.testsuite:
        fetch_testsuite(args.root, args.variants, args.max_file_mb)
    print("done. Try: datasleuth eval --config configs/zero_shot.yaml --limit 10")


if __name__ == "__main__":
    main()
