"""Get the Yelp Open Dataset files into a directory: from a URL, a local archive, or Kaggle.

Yelp distributes the dataset behind a licence form (https://business.yelp.com/data/resources/open-dataset/):
after accepting it you get a download link to an archive (``yelp_dataset.tar``, or ``.tgz``/``.zip``
depending on the release) that holds ``yelp_academic_dataset_*.json``, sometimes inside another
archive.  There is no stable public URL, so this module takes the link you were given
(``--url`` or ``YELP_DATASET_URL`` in ``.env``), an archive already on disk (``--archive``), or
the Kaggle mirror (``kaggle datasets download yelp-dataset/yelp-dataset``, needs the Kaggle CLI and
credentials), and extracts whatever it gets until the three JSON files are in place.

    python examples/yelp/fetch.py --data ~/yelp --url "https://...signed-link..."
    python examples/yelp/fetch.py --data ~/yelp --archive ~/Downloads/yelp_dataset.tar
    python examples/yelp/fetch.py --data ~/yelp --kaggle
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "business": "yelp_academic_dataset_business.json",
    "review": "yelp_academic_dataset_review.json",
    "user": "yelp_academic_dataset_user.json",
}
OTHER_FILES = ("yelp_academic_dataset_checkin.json", "yelp_academic_dataset_tip.json")
ARCHIVE_SUFFIXES = (".tar", ".tgz", ".tar.gz", ".zip")
KAGGLE_DATASET = "yelp-dataset/yelp-dataset"


def have_files(data: Path) -> bool:
    return all((data / name).is_file() for name in FILES.values())


def missing(data: Path) -> list[str]:
    return [name for name in FILES.values() if not (data / name).is_file()]


def ensure(data: Path, *, url: str | None = None, archive: Path | None = None, kaggle: bool = False, quiet: bool = False) -> Path:
    """Make sure the three JSON files are in ``data``; fetch and extract when they are not."""

    data = data.expanduser()
    data.mkdir(parents=True, exist_ok=True)
    if have_files(data):
        return data
    if archive is None and url is None and not kaggle:
        url = os.environ.get("YELP_DATASET_URL") or None
    if archive is not None:
        extract(Path(archive).expanduser(), data, quiet=quiet)
    elif url:
        extract(download(url, data, quiet=quiet), data, quiet=quiet)
    elif kaggle:
        extract(download_kaggle(data, quiet=quiet), data, quiet=quiet)
    else:
        raise SystemExit(
            f"The Yelp files are not in {data} (missing: {', '.join(missing(data))}).\n"
            "Get the dataset at https://business.yelp.com/data/resources/open-dataset/ (accept the licence, copy the download\n"
            "link), then either put YELP_DATASET_URL=<link> in .env, or pass --url <link>, --archive <downloaded file>, or --kaggle\n"
            f"(the Kaggle mirror {KAGGLE_DATASET}, with the kaggle CLI configured)."
        )
    if not have_files(data):
        raise SystemExit(f"Extracted the archive but the files are still missing in {data}: {', '.join(missing(data))}")
    return data


def download(url: str, data: Path, *, quiet: bool = False) -> Path:
    """Stream the archive into ``data`` (resumable: an existing complete file is reused)."""

    name = url.split("?")[0].rstrip("/").split("/")[-1] or "yelp_dataset.tar"
    if not name.endswith(ARCHIVE_SUFFIXES):
        name = "yelp_dataset.tar"
    target = data / name
    request = urllib.request.Request(url, headers={"User-Agent": "yodb-yelp-loader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        if target.is_file() and total and target.stat().st_size == total:
            log(f"already downloaded: {target}", quiet)
            return target
        log(f"downloading {url.split('?')[0]} ({total / 2**30:.1f} GB) to {target}", quiet)
        done, started, last = 0, time.perf_counter(), 0.0
        with target.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if not quiet and now - last > 5:
                    rate = done / max(now - started, 1e-9) / 2**20
                    log(f"  {done / 2**30:.2f} GB" + (f" of {total / 2**30:.2f} GB" if total else "") + f" ({rate:.0f} MB/s)", quiet)
                    last = now
    return target


def download_kaggle(data: Path, *, quiet: bool = False) -> Path:
    if shutil.which("kaggle") is None:
        raise SystemExit("the Kaggle CLI is not installed: pip install kaggle, and put your API token in ~/.kaggle/kaggle.json")
    log(f"downloading {KAGGLE_DATASET} from Kaggle into {data}", quiet)
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data)], check=True)
    archives = sorted(data.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not archives:
        raise SystemExit("Kaggle downloaded nothing")
    return archives[-1]


def extract(archive: Path, data: Path, *, quiet: bool = False, depth: int = 0) -> None:
    """Extract an archive into ``data``; archives found inside are extracted too (Yelp nests one)."""

    if depth > 2:
        return
    log(f"extracting {archive.name}", quiet)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _extract_member(handle.open(member), Path(member.filename).name, member.file_size, data, quiet)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle:
                if member.isfile():
                    _extract_member(handle.extractfile(member), Path(member.name).name, member.size, data, quiet)
    else:
        raise SystemExit(f"{archive} is not a tar or zip archive")
    for nested in list(data.iterdir()):
        if nested.is_file() and nested != archive and nested.name.endswith(ARCHIVE_SUFFIXES) and not have_files(data):
            extract(nested, data, quiet=quiet, depth=depth + 1)


def _extract_member(stream, name: str, size: int, data: Path, quiet: bool) -> None:
    """Keep the dataset files and nested archives, flat in ``data``; skip everything else (PDFs, folders)."""

    wanted = name in FILES.values() or name in OTHER_FILES or name.endswith(ARCHIVE_SUFFIXES)
    if not wanted or stream is None:
        return
    target = data / name
    if target.is_file() and target.stat().st_size == size:
        return
    log(f"  {name} ({size / 2**30:.2f} GB)", quiet)
    with target.open("wb") as handle:
        shutil.copyfileobj(stream, handle, 1 << 20)


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[fetch] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("YELP_DATA", "~/yelp")))
    parser.add_argument("--url", help="the download link from the Yelp form (or YELP_DATASET_URL in .env)")
    parser.add_argument("--archive", type=Path, help="an archive already downloaded")
    parser.add_argument("--kaggle", action="store_true", help=f"download {KAGGLE_DATASET} with the Kaggle CLI")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from yodb.dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    data = ensure(args.data, url=args.url, archive=args.archive, kaggle=args.kaggle, quiet=args.quiet)
    print(f"Yelp files ready in {data}: " + ", ".join(f"{name} ({(data / name).stat().st_size / 2**30:.2f} GB)" for name in FILES.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
