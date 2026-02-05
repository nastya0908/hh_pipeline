from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="HH level classifier PoC (junior/middle/senior)")
    p.add_argument("--input", required=True, help="Path to hh.csv")
    p.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    p.add_argument("--no-title", action="store_true", help="Do not use title in text features")
    args = p.parse_args()

    out_dir = str(Path(args.out))
    res = run_training(
        input_csv=args.input,
        out_dir=out_dir,
        include_title=not args.no_title,
    )

    print("=== DONE ===")
    print(f"Model saved: {res.model_path}")
    print(f"Artifacts dir: {out_dir}")
    print("Detected columns:")
    print(json.dumps(res.used_columns, ensure_ascii=False, indent=2))
    print("\nClassification report saved to artifacts/classification_report.txt")


if __name__ == "__main__":
    main()
