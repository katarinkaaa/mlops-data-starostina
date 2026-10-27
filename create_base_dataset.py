from clearml import Dataset

dataset = Dataset.create(
    dataset_name="sklearn-tabular-starostina-fixed",
    dataset_project="MIIT/MLOps/practice_3/starostina",
    dataset_version="1.0.0",
    description="Original sklearn Iris dataset",
)

dataset.add_files(path="clearml_source/base/data/iris.csv")
dataset.add_tags(["base"])
dataset.upload()
dataset_id = dataset.id
dataset.finalize()

print("BASE_ID =", dataset_id)

