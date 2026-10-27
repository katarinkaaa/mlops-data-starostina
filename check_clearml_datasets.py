
from clearml import Dataset

PROJECT = "MIIT/MLOps/practice_3/starostina"
NAME = "sklearn-tabular-starostina-fixed"

items = Dataset.list_datasets(
    dataset_project=PROJECT,
    partial_name=NAME,
    only_completed=True,
)

expected_versions = {
    "1.0.0": "base",
    "1.1.0": "wine",
    "2.0.0": "iris_features",
}

found = {}

for item in items:
    version = str(item.get("version", ""))
    if version in expected_versions:
        found[version] = item["id"]

for version, label in expected_versions.items():
    print("\n" + "=" * 45)
    print("Название:", label)
    print("Версия:", version)

    if version not in found:
        print("ОШИБКА: версия не найдена")
        continue

    dataset = Dataset.get(dataset_id=found[version])

    print("Статус:", dataset._task.status)
    print("Файлы:", dataset.list_files())
    print("Родители:", dataset._get_parents())

print("\nПроверка завершена.")
