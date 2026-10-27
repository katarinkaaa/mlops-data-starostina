
from pathlib import Path
import shutil
import pandas as pd

# Исходный Iris из DVC-версии data-v1
base_iris = Path("clearml_source/base/data/iris.csv")

# Папки для корректных источников
base_dir = Path("clearml_source/base")
wine_dir = Path("clearml_source/wine")
features_dir = Path("clearml_source/iris_features")

wine_dir.mkdir(parents=True, exist_ok=True)
features_dir.mkdir(parents=True, exist_ok=True)

# 1. Base: исходный Iris, 5 столбцов
base_df = pd.read_csv(base_iris)
assert base_df.shape == (150, 5), f"Неверный размер base: {base_df.shape}"

# 2. Wine: такой же исходный Iris, 5 столбцов
shutil.copy2(base_iris, wine_dir / "iris.csv")

# 3. Iris features: Iris с двумя дополнительными признаками
features_df = base_df.copy()
features_df["sepal_area"] = (
    features_df["sepal length (cm)"] * features_df["sepal width (cm)"]
)
features_df["petal_area"] = (
    features_df["petal length (cm)"] * features_df["petal width (cm)"]
)
features_df.to_csv(features_dir / "iris.csv", index=False)

print("base:", pd.read_csv(base_dir / "data/iris.csv").shape)
print("wine Iris:", pd.read_csv(wine_dir / "iris.csv").shape)
print("iris_features:", pd.read_csv(features_dir / "iris.csv").shape)
print("Столбцы iris_features:", pd.read_csv(features_dir / "iris.csv").columns.tolist())
