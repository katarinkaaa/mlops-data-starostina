from clearml import Dataset

BASE_ID = "295751f06c1f4b1cbd29edf8165bea87"

dataset = Dataset.create(
    dataset_name="sklearn-tabular-starostina-fixed",
    dataset_project="MIIT/MLOps/practice_3/starostina",
    dataset_version="2.0.0",
    description="Iris dataset with sepal_area and petal_area features",
    parent_datasets=[BASE_ID],
)

dataset.add_files(path="clearml_source/iris_features/iris.csv")
dataset.add_tags(["iris_features"])
dataset.upload()

dataset_id = dataset.id
dataset.finalize()

print("IRIS_FEATURES_ID =", dataset_id)