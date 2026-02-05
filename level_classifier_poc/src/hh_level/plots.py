from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_class_balance(y: pd.Series, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    counts = y.value_counts().sort_index()

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class balance: junior/middle/senior")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
