from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", sep=",")
    df = df.rename(columns=lambda c: c.replace("\ufeff", ""))
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.replace({"\u00a0": " "}, regex=True)
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def parse_salary(df: pd.DataFrame) -> pd.DataFrame:
    salary_column_name = "ЗП"
    df[salary_column_name] = df[salary_column_name].astype(str).str.lower().str.replace(" ", "", regex=False)
    
    currency_rates = {"руб.": 1.0, "руб": 1.0, "kzt": 0.151579, "usd": 83.0}
    
    def convert_salary_to_rub(value: str) -> float:
        for code, rate in currency_rates.items():
            if code in value:
                numeric_part = re.sub(r'[^0-9.]', '', value)
                try:
                    return float(numeric_part) * rate
                except:
                    return np.nan
        return np.nan
    
    df["salary_rub"] = df[salary_column_name].map(convert_salary_to_rub)
    return df


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    salary_series = df["salary_rub"].fillna(0)
    q1, q3 = salary_series.quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (salary_series >= q1 - 1.5*iqr) & (salary_series <= q3 + 1.5*iqr)
    return df[mask].reset_index(drop=True)


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df["sex"] = df["Пол, возраст"].str.contains("мужчина", case=False, na=False).astype(np.float32)
    
    def get_age(text):
        match = re.search(r'(\d+)\s*(?:год|лет)', str(text).lower())
        return float(match.group(1)) if match else 30.0
    
    df["age"] = df["Пол, возраст"].map(get_age)
    df["education"] = df["Образование и ВУЗ"].str.contains("высшее", case=False, na=False).astype(np.float32)
    
    y = df["salary_rub"].fillna(df["salary_rub"].median()).astype(np.float32)
    x = df[["sex", "age", "education"]].astype(np.float32).values
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    
    return x, y


def main(csv_path_str: str):
    csv_path = Path(csv_path_str).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")
    
    print("1. Загрузка CSV...")
    df = load_csv(csv_path)
    
    print("2. Парсинг зарплаты...")
    df = parse_salary(df)
    
    print("3. Очистка выбросов...")
    df = clean_outliers(df)
    
    print("4. Извлечение признаков...")
    x, y = extract_features(df)
    
    x_path = csv_path.parent / "x_data.npy"
    y_path = csv_path.parent / "y_data.npy"
    
    np.save(x_path, x)
    np.save(y_path, y)
    
    print(f"✅ x_data.npy: {x_path} ({x.nbytes/1024/1024:.1f}MB)")
    print(f"✅ y_data.npy: {y_path} ({y.nbytes/1024/1024:.1f}MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python app_fixed.py path/to/hh.csv")
        sys.exit(1)
    main(sys.argv[1])
