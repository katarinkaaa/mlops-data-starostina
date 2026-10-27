
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_file(path, expected_rows):
    if not path.exists():
        raise FileNotFoundError(f"Нет файла: {path}")

    df = pd.read_csv(path)

    if any(col.startswith("Unnamed:") for col in df.columns):
        raise ValueError(f"{path}: обнаружен лишний индексный столбец")

    if len(df) != expected_rows:
        raise ValueError(
            f"{path}: ожидалось {expected_rows} строк, получено {len(df)}"
        )

    print(f"\nФайл: {path}")
    print(f"SHA-256: {sha256(path)}")
    print(f"Размер: {df.shape}")
    print(f"Столбцы: {df.columns.tolist()}")
    print(f"Пропуски: {df.isna().sum().sum()}")

    return df


def main():
    iris_path = DATA_DIR / "iris.csv"
    wine_path = DATA_DIR / "wine.csv"

    iris_original = load_iris(as_frame=True).frame
    wine_original = load_wine(as_frame=True).frame

    iris = check_file(iris_path, len(iris_original))

    if "sepal_area" in iris.columns or "petal_area" in iris.columns:
        expected_sepal = (
            iris["sepal length (cm)"] * iris["sepal width (cm)"]
        )
        expected_petal = (
            iris["petal length (cm)"] * iris["petal width (cm)"]
        )

        if "sepal_area" not in iris.columns or "petal_area" not in iris.columns:
            raise ValueError("Не хватает одного из новых признаков")

        if not (
            (iris["sepal_area"] - expected_sepal).abs() < 1e-8
        ).all():
            raise ValueError("Неверные значения sepal_area")

        if not (
            (iris["petal_area"] - expected_petal).abs() < 1e-8
        ).all():
            raise ValueError("Неверные значения petal_area")

    if wine_path.exists():
        check_file(wine_path, len(wine_original))

    print("\nПроверка завершена.")


if __name__ == "__main__":
    main()
