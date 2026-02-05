from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_dataframe(
    df: pd.DataFrame,
    col_title: str | None,
    col_skills: str | None,
    col_desc: str | None,
    col_salary: str | None,
    col_city: str | None,
    col_age: str | None,
    col_exp_years: str,
    include_title: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    def sget(col: str | None):
        if col is None or col not in out.columns:
            return ""
        return out[col].fillna("").astype(str)

    title = sget(col_title) if include_title else ""
    skills = sget(col_skills)
    desc = sget(col_desc)

    out["text"] = (
        (title + " " + skills + " " + desc)
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if col_salary is not None and col_salary in out.columns:
        out["salary_num"] = pd.to_numeric(out[col_salary], errors="coerce")
    else:
        out["salary_num"] = np.nan

    if col_age is not None and col_age in out.columns:
        out["age_num"] = pd.to_numeric(out[col_age], errors="coerce")
    else:
        out["age_num"] = np.nan

    out["exp_years"] = pd.to_numeric(out[col_exp_years], errors="coerce")

    if col_city is not None and col_city in out.columns:
        out["city_cat"] = out[col_city].fillna("unknown").astype(str)
    else:
        out["city_cat"] = "unknown"

    out["salary_log"] = np.log1p(out["salary_num"].clip(lower=0))

    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = ["salary_log", "age_num", "exp_years"]
    categorical = ["city_cat"]
    text = "text"

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    txt = TfidfVectorizer(min_df=2, ngram_range=(1, 2))

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
            ("txt", txt, text),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_model_pipeline(model) -> Pipeline:
    pre = build_preprocessor()
    return Pipeline([("pre", pre), ("clf", model)])
