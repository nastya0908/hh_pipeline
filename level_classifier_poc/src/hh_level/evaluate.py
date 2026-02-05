from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def save_report(y_true, y_pred, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_txt = classification_report(y_true, y_pred, digits=3)
    (out / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        out / "confusion_matrix.csv", encoding="utf-8"
    )

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_dict["n_samples"] = int(len(y_true))
    report_dict["labels"] = labels

    (out / "metrics.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_dict
