from pathlib import Path
from urllib.parse import urlparse

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "source_registry.yaml"


def main() -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file)

    if not isinstance(registry, dict):
        raise ValueError("Registry must be a YAML mapping.")

    sources = registry.get("sources", [])

    if not sources:
        raise ValueError("No sources found.")

    required_fields = {
        "id",
        "domain",
        "source_type",
        "authority",
        "title",
        "url",
    }

    source_ids = set()

    for source in sources:
        missing = required_fields - source.keys()

        if missing:
            raise ValueError(
                f"Source {source.get('id', '<unknown>')} "
                f"is missing: {sorted(missing)}"
            )

        source_id = source["id"]

        if source_id in source_ids:
            raise ValueError(f"Duplicate source ID: {source_id}")

        source_ids.add(source_id)

        parsed = urlparse(source["url"])

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Invalid URL for {source_id}: {source['url']}"
            )

    print("=" * 60)
    print("DSQA Source Registry Validation")
    print("=" * 60)
    print(f"Registry : {REGISTRY_PATH}")
    print(f"Sources  : {len(sources)}")
    print("-" * 60)

    for source in sources:
        print(
            f"[OK] {source['id']} "
            f"| {source['domain']} "
            f"| {source['authority']}"
        )

    print("-" * 60)
    print("Source registry validation: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()