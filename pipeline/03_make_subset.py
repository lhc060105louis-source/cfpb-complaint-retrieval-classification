from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import get_logger


@dataclass
class SubsetSpec:
    src: Path
    dst: Path
    n_lines: int

    @classmethod
    def from_env(cls) -> "SubsetSpec":
        return cls(
            src=Path(os.environ.get("CS410_SRC_PATH", "data/complaints.csv")),
            dst=Path(os.environ.get("CS410_SUBSET_PATH", "data/complaints_subset.csv")),
            n_lines=int(os.environ.get("CS410_N_LINES", "200001")),
        )


def materialize_subset(spec: SubsetSpec) -> int:
    if not spec.src.exists():
        raise FileNotFoundError(f"source csv not found: {spec.src}")
    spec.dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with spec.src.open("r", encoding="utf-8", errors="ignore", newline="") as fin, spec.dst.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        for _ in range(spec.n_lines):
            line = fin.readline()
            if not line:
                break
            fout.write(line)
            written += 1
    return written


def main() -> int:
    log = get_logger("03_make_subset")
    spec = SubsetSpec.from_env()
    written = materialize_subset(spec)
    log.info("wrote %d lines to %s", written, spec.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
