
import argparse
from pathlib import Path

from sklearn.datasets import load_iris, load_wine


DATA_DIR = Path("data")


def save_csv(df, path):
    df = df.sort_values(by=list(df.columns), kind="mergesort")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.10g",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["base", "wine", "iris_features"],
    )
    mode = parser.parse_args().mode

    if mode == "base":
        save_csv(load_iris(as_frame=True).frame, DATA_DIR / "iris.csv")

    elif mode == "wine":
        save_csv(load_wine(as_frame=True).frame, DATA_DIR / "wine.csv")

    elif mode == "iris_features":
        iris = load_iris(as_frame=True).frame
        iris["sepal_area"] = (
            iris["sepal length (cm)"] * iris["sepal width (cm)"]
        )
        iris["petal_area"] = (
            iris["petal length (cm)"] * iris["petal width (cm)"]
        )
        save_csv(iris, DATA_DIR / "iris.csv")


if __name__ == "__main__":
    main()
