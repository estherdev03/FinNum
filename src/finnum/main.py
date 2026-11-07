from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

TRAIN_DATA_URL = (
    "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json"
)

data = load_dataset("json", data_files={"train": TRAIN_DATA_URL})
df = pd.DataFrame(data["train"])

print("Shape: ", df.shape)
print("Columns: ", df.columns)
print("Info: ", df.info())
for column in df.columns.to_list():
    print("-------Column %s ----------" % column)
    print(df.loc[0, column])
    print("========================")
