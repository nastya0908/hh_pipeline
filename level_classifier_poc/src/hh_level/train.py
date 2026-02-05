from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .evaluate import save_report
from .features import build_model_pipeline, prepare_dataframe
from .filters import is_it_developer
from .io import detect_columns, read_csv, to_numeric_safe
from .labeling import make_label, parse_age_from_text, parse_experience_years
from .plots import plot_class_balance


@dataclass
class TrainResult:
    model_path: str
    report: dict
    used_columns: dict


def run_training(
    input_csv: str,
    out_dir: str,
    include_title: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    df = read_csv(input_csv)
    cols = detect_columns(df)

    col_title = cols["title"]
    col_skills = cols["skills"]
    col_desc = cols["description"]
    col_salary = cols["salary"]
    col_city = cols["city"]
    col_age = cols["age"]
    col_exp = cols["experience"]

    if col_title is None:
        raise ValueError("Не нашёл колонку с названием должности (title/position/...).")

    # Фильтр dev
    df["__skills_tmp__"] = df[col_skills] if col_skills in df.columns else ""
    dev_mask = df.apply(lambda r: is_it_developer(r[col_title], r["__skills_tmp__"]), axis=1)
    df_dev = df[dev_mask].copy()

    # Опыт -> годы
    if col_exp is None:
        df_dev["__exp_years__"] = float("nan")
    else:
        df_dev["__exp_years__"] = df_dev[col_exp].apply(parse_experience_years)

    # y
    df_dev["y"] = df_dev.apply(lambda r: make_label(r[col_title], r["__exp_years__"]), axis=1)
    df_dev = df_dev[df_dev["y"].notna()].copy()

    # salary numeric
    if col_salary is not None and col_salary in df_dev.columns:
        df_dev[col_salary] = df_dev[col_salary].apply(to_numeric_safe)

    # age numeric: если "Пол, возраст" — парсим возраст из текста
    if col_age is not None and col_age in df_dev.columns:
        if str(col_age).strip().lower() in ["пол, возраст", "пол,возраст", "пол возраст"]:
            df_dev["__age_num__"] = df_dev[col_age].apply(parse_age_from_text)
            col_age_used = "__age_num__"
        else:
            df_dev[col_age] = df_dev[col_age].apply(to_numeric_safe)
            col_age_used = col_age
    else:
        col_age_used = None

    # График баланса
    plot_class_balance(df_dev["y"], f"{out_dir}/class_balance.png")

    prepared = prepare_dataframe(
        df_dev,
        col_title=col_title,
        col_skills=col_skills,
        col_desc=col_desc,
        col_salary=col_salary,
        col_city=col_city,
        col_age=col_age_used,
        col_exp_years="__exp_years__",
        include_title=include_title,
    )

    X = prepared[["salary_log", "age_num", "exp_years", "city_cat", "text"]]
    y = df_dev["y"].astype(str)

    # если классов мало или один класс — не обучаем
    if y.nunique() < 2:
        raise ValueError(
            f"Слишком мало классов после фильтрации/разметки: {y.value_counts().to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )

    pipe = build_model_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = save_report(y_test, y_pred, out_dir)

    model_path = f"{out_dir}/model.joblib"
    joblib.dump(pipe, model_path)

    return TrainResult(model_path=model_path, report=report, used_columns=cols)
