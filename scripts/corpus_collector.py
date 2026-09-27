from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = PROJECT_ROOT / "configs" / "source_registry.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOCUMENTS_DIR = RAW_DIR / "documents"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"

USER_AGENT = (
    "IEEE-CIS-DSQA-2026-Research-Corpus-Collector/0.1 "
    "(pre-challenge research)"
)

REQUEST_TIMEOUT = (10, 30)
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # 25 MB per source


def load_registry() -> dict:
    """Load the public-source registry."""
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file)

    if not isinstance(registry, dict):
        raise ValueError("Source registry must be a YAML mapping.")

    return registry


def validate_url(url: str) -> None:
    """Validate that a source uses HTTP or HTTPS."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {url}")

    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")


def safe_filename(source: dict, content_type: str | None = None) -> str:
    """Create a stable filename for a source."""
    source_id = source["id"]

    parsed = urlparse(source["url"])
    suffix = Path(parsed.path).suffix.lower()

    if suffix not in {".html", ".htm", ".pdf"}:
        guessed = mimetypes.guess_extension(
            (content_type or "").split(";")[0].strip()
        )

        if guessed in {".html", ".htm", ".pdf"}:
            suffix = guessed
        else:
            suffix = ".bin"

    return f"{source_id}{suffix}"


def sha256_bytes(data: bytes) -> str:
    """Calculate SHA-256 for downloaded content."""
    return hashlib.sha256(data).hexdigest()


def is_allowed_content_type(content_type: str | None) -> bool:
    """Allow common HTML/PDF document responses."""
    if not content_type:
        return True

    media_type = content_type.split(";")[0].strip().lower()

    return media_type in {
        "text/html",
        "application/xhtml+xml",
        "application/pdf",
        "application/octet-stream",
    }


def collect_source(source: dict, dry_run: bool) -> dict:
    """Collect one public source and return manifest metadata."""
    source_id = source["id"]
    url = source["url"]

    validate_url(url)

    retrieved_at = datetime.now(timezone.utc).isoformat()

    result = {
        "source_id": source_id,
        "domain": source["domain"],
        "title": source["title"],
        "url": url,
        "source_type": source["source_type"],
        "authority": source["authority"],
        "retrieved_at": retrieved_at,
        "status": "dry_run" if dry_run else "pending",
    }

    if dry_run:
        return result

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        stream=True,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type")

    if not is_allowed_content_type(content_type):
        raise ValueError(
            f"Unsupported content type for {source_id}: {content_type}"
        )

    content_length = response.headers.get("Content-Length")

    if content_length:
        try:
            declared_size = int(content_length)

            if declared_size > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Response too large for {source_id}: "
                    f"{declared_size} bytes"
                )
        except ValueError as exc:
            if "Response too large" in str(exc):
                raise

    chunks: list[bytes] = []
    total_size = 0

    for chunk in response.iter_content(chunk_size=1024 * 256):
        if not chunk:
            continue

        total_size += len(chunk)

        if total_size > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Response exceeded {MAX_DOWNLOAD_BYTES} bytes: {source_id}"
            )

        chunks.append(chunk)

    content = b"".join(chunks)

    if not content:
        raise ValueError(f"Empty response from {source_id}")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(source, content_type)
    local_path = DOCUMENTS_DIR / filename

    local_path.write_bytes(content)

    content_hash = sha256_bytes(content)

    result.update(
        {
            "status": "collected",
            "http_status": response.status_code,
            "final_url": response.url,
            "content_type": content_type,
            "content_bytes": len(content),
            "content_hash": content_hash,
            "local_path": str(local_path.relative_to(PROJECT_ROOT)),
        }
    )

    return result


def write_manifest(records: list[dict]) -> None:
    """Write collection metadata as JSON Lines."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def get_enabled_sources(registry: dict) -> list[dict]:
    """Return enabled sources from the registry."""
    sources = registry.get("sources", [])

    if not isinstance(sources, list):
        raise ValueError("Registry 'sources' must be a list.")

    return [
        source
        for source in sources
        if isinstance(source, dict) and source.get("enabled", True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect public sources registered for the "
            "IEEE CIS DSQA 2026 pre-challenge corpus."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview collection without downloading.",
    )

    args = parser.parse_args()

    registry = load_registry()
    sources = get_enabled_sources(registry)

    if not sources:
        raise ValueError("No enabled sources found.")

    print("=" * 70)
    print("IEEE CIS DSQA 2026 — Public Corpus Collector")
    print("=" * 70)
    print(f"Registry : {REGISTRY_PATH}")
    print(f"Sources  : {len(sources)}")
    print(
        "Mode     : "
        + ("DRY-RUN (no downloads)" if args.dry_run else "COLLECT")
    )
    print("-" * 70)

    records: list[dict] = []

    for index, source in enumerate(sources, start=1):
        source_id = source["id"]

        print(
            f"[{index:02d}/{len(sources):02d}] "
            f"{source_id} | {source['domain']}"
        )
        print(f"       URL: {source['url']}")

        try:
            record = collect_source(
                source,
                dry_run=args.dry_run,
            )

            records.append(record)

            if args.dry_run:
                print("       Result: READY")
            else:
                print(
                    f"       Result: OK "
                    f"({record['content_bytes']} bytes)"
                )

        except Exception as exc:
            error_record = {
                "source_id": source_id,
                "domain": source.get("domain"),
                "title": source.get("title"),
                "url": source.get("url"),
                "source_type": source.get("source_type"),
                "authority": source.get("authority"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": str(exc),
            }

            records.append(error_record)

            print(f"       Result: ERROR — {exc}")

    write_manifest(records)

    success_count = sum(
        record["status"] in {"dry_run", "collected"}
        for record in records
    )

    error_count = sum(
        record["status"] == "error"
        for record in records
    )

    print("-" * 70)
    print(f"Successful : {success_count}")
    print(f"Errors     : {error_count}")
    print(f"Manifest   : {MANIFEST_PATH}")

    if args.dry_run:
        print("Download   : SKIPPED")
    else:
        print(f"Documents  : {DOCUMENTS_DIR}")

    print("-" * 70)

    if error_count:
        print("Corpus collection completed with errors.")
        sys.exit(1)

    print("Corpus collector: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()