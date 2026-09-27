import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def main() -> None:
    config = load_config()

    print("=" * 60)
    print("IEEE CIS DSQA 2026 — Environment Smoke Test")
    print("=" * 60)

    print(f"Python : {sys.version.split()[0]}")
    print(f"Root   : {PROJECT_ROOT}")
    print(f"Config : {config['project']['name']}")
    print(f"Model constraint : {config['challenge']['constraints']['fixed_model']}")
    print(f"Max runtime      : {config['challenge']['constraints']['max_wall_clock_minutes']} min")
    print(f"Submission limit : {config['challenge']['constraints']['max_submission_size_mb']} MB")

    print("-" * 60)
    print("Smoke test: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()