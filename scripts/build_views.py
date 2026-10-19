"""Build docs/views/index.html from the saved results in docs/results/.

Each `el` command writes its numbers there with `--results-dir docs/results`. This script only reads them, so the
page rebuilds in a second and never reruns a model. Run `uv run python scripts/build_views.py`.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ROOT / "docs" / "views"
RESULTS = ROOT / "docs" / "results"
PLACEHOLDER = "/*VIEW_DATA*/null"
COMMANDS = ["pooling", "similarity", "ann", "finetune", "dims", "chunks"]


def build(results: Path = RESULTS) -> dict:
    missing = [name for name in COMMANDS if not (results / f"{name}.json").exists()]
    if missing:
        raise SystemExit(f"missing results for {missing}; run `uv run el --results-dir docs/results <command>`")
    return {name: json.loads((results / f"{name}.json").read_text()) for name in COMMANDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=VIEWS / "index.html")
    args = parser.parse_args()
    template = (VIEWS / "template.html").read_text()
    if PLACEHOLDER not in template:
        raise SystemExit(f"{PLACEHOLDER} is missing from the template")
    args.out.write_text(template.replace(PLACEHOLDER, json.dumps(build())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
