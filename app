#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    args = parser.parse_args()

    x_path = args.x_path.resolve()
    if not x_path.exists():
        raise FileNotFoundError(f"Файл не найден: {x_path}")

    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "resources" / "salary_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path} (сначала запусти train_model.py)")

    model = joblib.load(model_path)
    x = np.load(x_path).astype(np.float32)
    y_pred = model.predict(x)

    print(list(map(float, y_pred.tolist())))


if __name__ == "__main__":
    main()
