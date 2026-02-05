from __future__ import annotations

import re
from typing import Optional

import numpy as np

LEVELS = ("junior", "middle", "senior")


def normalize_text(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("джун", "junior").replace("мидл", "middle").replace("сеньор", "senior")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_age_from_text(value: str | None) -> float:
    """
    Из колонки 'Пол, возраст' вытаскиваем возраст: 'Мужчина, 42 года ...' -> 42
    """
    if value is None:
        return np.nan
    s = normalize_text(value)
    m = re.search(r"(\d{1,2})\s*(год|года|лет)", s)
    if not m:
        return np.nan
    age = float(m.group(1))
    if age < 14 or age > 90:
        return np.nan
    return age


def label_from_title(title: str | None) -> Optional[str]:
    t = normalize_text(title)
    if not t:
        return None

    if any(k in t for k in ["intern", "trainee", "entry", "junior", "стажер", "стажёр", "младший"]):
        return "junior"

    if any(
        k in t
        for k in [
            "lead",
            "team lead",
            "tech lead",
            "principal",
            "staff",
            "senior",
            "старший",
            "ведущий",
            "главный",
            "руководитель",
        ]
    ):
        return "senior"

    if "middle" in t or "mid-level" in t or "миддл" in t:
        return "middle"

    return None


def parse_experience_years(exp) -> float:
    """
    Поддерживаем:
    - число (считаем годы)
    - строки "6 лет 5 месяцев", "19 лет", "18 months"
    """
    if exp is None:
        return np.nan
    if isinstance(exp, (int, float)) and not np.isnan(exp):
        return float(exp)

    s = normalize_text(str(exp))
    if not s:
        return np.nan

    # если просто число
    m = re.fullmatch(r"[-+]?\d+(\.\d+)?", s)
    if m:
        return float(s)

    years = 0.0
    months = 0.0

    # русские варианты
    my = re.search(r"(\d+(\.\d+)?)\s*(год|года|лет)", s)
    if my:
        years = float(my.group(1))
    mm = re.search(r"(\d+(\.\d+)?)\s*(месяц|месяца|месяцев)", s)
    if mm:
        months = float(mm.group(1))

    # англ варианты
    my2 = re.search(r"(\d+(\.\d+)?)\s*(year|years|yr|yrs)", s)
    if my2:
        years = float(my2.group(1))
    mm2 = re.search(r"(\d+(\.\d+)?)\s*(month|months|mo|mos)", s)
    if mm2:
        months = float(mm2.group(1))

    total = years + months / 12.0
    if total <= 0:
        m_any = re.search(r"(\d+(\.\d+)?)", s)
        if m_any:
            return float(m_any.group(1))
        return np.nan
    return total


def label_from_experience(exp_years: float) -> Optional[str]:
    if exp_years is None or (isinstance(exp_years, float) and np.isnan(exp_years)):
        return None
    if exp_years < 1.5:
        return "junior"
    if exp_years < 4.5:
        return "middle"
    return "senior"


def make_label(title: str | None, exp_years: float) -> Optional[str]:
    lt = label_from_title(title)
    if lt is not None:
        return lt
    return label_from_experience(exp_years)
