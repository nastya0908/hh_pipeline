from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    x_path = repo_root / "x_data.npy"
    y_path = repo_root / "y_data.npy"
    resources_dir = repo_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("MAE:", float(mean_absolute_error(y_test, y_pred)))
    print("R2:", float(r2_score(y_test, y_pred)))

    joblib.dump(model, resources_dir / "salary_model.joblib")


if __name__ == "__main__":
    main()
