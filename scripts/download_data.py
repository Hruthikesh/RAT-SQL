"""Obtain datasets.

Spider is licensed CC BY-SA 4.0 and is *not* bundled with this repository.
This script downloads the official archive from the link published on the
Spider website (https://yale-lily.github.io/spider), extracts it into
``data/raw/`` and validates the layout.

Usage::

    python scripts/download_data.py --dataset synthetic     # generate the toy dataset
    python scripts/download_data.py --dataset spider        # download + extract + validate Spider
    python scripts/download_data.py --dataset spider --validate-only
    python scripts/download_data.py --official-eval         # fetch the official evaluation scripts
                                                           # (third_party/, for cross-checking only)

If the automatic download fails (Google Drive quotas), download
``spider_data.zip`` manually from the Spider website and run::

    python scripts/download_data.py --dataset spider --zip path/to/spider_data.zip
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ratsql.data.spider import find_spider_root, spider_spec, synthetic_spec  # noqa: E402
from ratsql.data.synthetic import generate_synthetic_dataset  # noqa: E402
from ratsql.data.validate import format_report, validate_dataset  # noqa: E402

SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"  # official link on yale-lily.github.io/spider
OFFICIAL_EVAL_FILES = {
    # Official Spider evaluation (test-suite version, Zhong et al. 2020); used only for cross-checking.
    "evaluation.py": "https://raw.githubusercontent.com/taoyds/test-suite-sql-eval/master/evaluation.py",
    "process_sql.py": "https://raw.githubusercontent.com/taoyds/test-suite-sql-eval/master/process_sql.py",
    "exec_eval.py": "https://raw.githubusercontent.com/taoyds/test-suite-sql-eval/master/exec_eval.py",
    "parse.py": "https://raw.githubusercontent.com/taoyds/test-suite-sql-eval/master/parse.py",
}


def download_spider(raw_dir: Path, zip_path: Path | None) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        zip_path = raw_dir / "spider_data.zip"
        if not zip_path.exists():
            try:
                import gdown  # type: ignore
            except ImportError:
                sys.exit("gdown is required for the automatic download: pip install gdown (or pass --zip).")
            print(f"Downloading Spider from Google Drive id {SPIDER_GDRIVE_ID} ...")
            out = gdown.download(id=SPIDER_GDRIVE_ID, output=str(zip_path), quiet=False)
            if out is None or not zip_path.exists():
                sys.exit("Download failed. Download spider_data.zip manually and pass --zip.")
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        zf.extractall(raw_dir, members)
    root = find_spider_root(raw_dir)
    if root is None:
        sys.exit(f"Could not find tables.json/dev.json after extracting into {raw_dir}")
    print(f"Spider root: {root}")
    return root


def fetch_official_eval(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in OFFICIAL_EVAL_FILES.items():
        if (dest / name).exists():
            continue
        for attempt in range(4):
            try:
                print(f"Fetching {url}")
                urllib.request.urlretrieve(url, dest / name)
                break
            except OSError as e:  # transient network errors
                print(f"  attempt {attempt + 1} failed: {e}")
        else:
            sys.exit(f"could not download {url}")
    (dest / "README.md").write_text(
        "Official Spider evaluation scripts (test-suite-sql-eval, Apache-2.0),\n"
        "downloaded by scripts/download_data.py --official-eval. Used only to cross-check\n"
        "our independent exact-match implementation; not part of this project's code.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["spider", "synthetic", "all", "none"], default="all")
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--zip", default=None, help="use an already-downloaded spider_data.zip")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--official-eval", action="store_true", help="download official Spider evaluation scripts into third_party/")
    args = ap.parse_args()

    if args.dataset in ("synthetic", "all"):
        out = ROOT / "data" / "samples" / "synthetic"
        info = generate_synthetic_dataset(out)
        print(f"Synthetic dataset written to {out}: {info}")
        print(format_report(validate_dataset(synthetic_spec(out))))
    if args.dataset in ("spider", "all"):
        raw = Path(args.raw_dir)
        root = find_spider_root(raw)
        if root is None and not args.validate_only:
            root = download_spider(raw, Path(args.zip) if args.zip else None)
        if root is None:
            sys.exit(f"Spider not found under {raw}")
        report = validate_dataset(spider_spec(root))
        print(format_report(report))
        if not report["ok"]:
            sys.exit(1)
    if args.official_eval:
        fetch_official_eval(ROOT / "third_party" / "spider_eval")


if __name__ == "__main__":
    main()
