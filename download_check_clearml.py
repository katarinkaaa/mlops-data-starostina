
from clearml import Dataset
from pathlib import Path
import hashlib
import pandas as pd

BASE_ID = "e1a564ebe4994e7dbabdc3974855e89f"
WINE_ID = "d601d24710904483bb3c36f2a726bd81"
IRIS_FEATURES_ID = "ad693ca704454d5e9738972f9ee400dd"

datasets = {
    "base": BASE_ID,
    "wine": WINE_ID,
    "iris_features": IRIS_FEATURES_ID,
}

output_root = Path("clearml_downloads")
output_root.mkdir(exist_ok=True)

for name, dataset_id in datasets.items():
    print(f"\n{'=' * 60}")
    print("ВЕРСИЯ:", name)

    dataset = Dataset.get(dataset_id=dataset_id)
    target_dir = output_root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset.get_mutable_local_copy(
        target_folder=str(target_dir)
    )

    for file_path in sorted(target_dir.rglob("*.csv")):
        if not file_path.is_file():
            continue

        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        df = pd.read_csv(file_path)

        print(f"\nФайл: {file_path.relative_to(output_root)}")
        print("SHA-256:", digest)
        print("Размер таблицы:", df.shape)
        print("Столбцы:", df.columns.tolist())
        print("Пропусков:", int(df.isna().sum().sum()))
