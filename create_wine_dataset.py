from clearml import Dataset

BASE_ID = "295751f06c1f4b1cbd29edf8165bea87"

dataset = Dataset.create(
    dataset_name="sklearn-tabular-starostina-fixed",
    dataset_project="MIIT/MLOps/practice_3/starostina",
    dataset_version="1.1.0",
    description="Base Iris dataset with added Wine dataset",
    parent_datasets=[BASE_ID],
)

dataset.add_files(path="clearml_source/wine/iris.csv")
dataset.add_files(path="clearml_source/wine/wine.csv")
dataset.add_tags(["wine"])
dataset.upload()

dataset_id = dataset.id
dataset.finalize()

print("WINE_ID =", dataset_id)
