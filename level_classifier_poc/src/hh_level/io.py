from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_COL_CANDIDATES


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    mapping: dict[str, Optional[str]] = {}
    for key, candidates in DEFAULT_COL_CANDIDATES.items():
        mapping[key] = _find_column(df, candidates)
    return mapping


def read_csv(path: str) -> pd.DataFrame:
    """
    Устойчивое чтение CSV для PoC:
    - python engine терпимее к битым кавычкам
    - sep=None -> автоопределение разделителя (только python engine)
    - on_bad_lines="skip" -> пропускаем битые строки
    """
    # pandas>=2: есть encoding_errors
    try:
        return pd.read_csv(
            path,
            engine="python",
            sep=None,
            on_bad_lines="skip",
            encoding="utf-8",
            encoding_errors="ignore",
        )
    except TypeError:
        # на случай более старого API без encoding_errors
        return pd.read_csv(
            path,
            engine="python",
            sep=None,
            on_bad_lines="skip",
            encoding="utf-8",
        )
    except Exception:
        # fallback: попробуем явно ';' и ','
        for sep in [";", ",", "\t"]:
            try:
                return pd.read_csv(
                    path,
                    engine="python",
                    sep=sep,
                    on_bad_lines="skip",
                    encoding="utf-8",
                    encoding_errors="ignore",
                )
            except Exception:
                continue
        # последний шанс: latin-1
        return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip", encoding="latin-1")


def to_numeric_safe(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    s = s.replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan
