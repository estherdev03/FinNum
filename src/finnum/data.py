# Load all train/dev/test data
# train: used to train the model
# dev: used to evaluate the model to adjust weights and hyperparameters
# test: main evaluation set

from datasets import load_dataset
import pandas as pd
from os.path import join

# Base url of data
BASE_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"

# Load 3 datasets
data = load_dataset(
    "json",
    data_files={
        "train": join(BASE_URL, "train.json"),
        "test": join(BASE_URL, "test.json"),
        "dev": join(BASE_URL, "dev.json"),
    },
)

# Turn each dataset into dataframe
train_df = pd.DataFrame(data["train"])
test_df = pd.DataFrame(data["test"])
dev_df = pd.DataFrame(data["dev"])

# Clean up all the columns, keep only what are needed
COLUMNS_KEPT = ["id", "filename", "pre_text", "post_text", "table", "qa"]
train_df = train_df.filter(items=COLUMNS_KEPT)
test_df = test_df.filter(items=COLUMNS_KEPT)
dev_df = dev_df.filter(items=COLUMNS_KEPT)

# Get the necessary columns in QA, flatten the data
for k in ["question", "program", "exe_ans"]:
    train_df[k] = train_df["qa"].str.get(k)
    test_df[k] = test_df["qa"].str.get(k)
    dev_df[k] = dev_df["qa"].str.get(k)

# Remove the QA col
train_df = train_df.drop(columns="qa")
test_df = test_df.drop(columns="qa")
dev_df = dev_df.drop(columns="qa")
