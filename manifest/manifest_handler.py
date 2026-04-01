import pandas as pd
import os

# Path for manfiest.csv file containing the dataset
# The os methods ensure that the script locates manifest.csv correctly 
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.csv")

df = pd.DataFrame()

# Load dataset to a Pandas Dataframe for changes
def load_dataset():
    global df
    df = pd.read_csv(MANIFEST_PATH, dtype={"image_path": "str", "target_class": "int32", "model": "str", "noise_path": "str"})

# Add a new entry to the dataset
def add_entry(image_path: str, target_class: int, model: str, noise_path: str) -> None:
    if (df.empty): load_dataset()
    df.loc[df.shape[0]] = {"":df.shape[0], "image_path": image_path, "target_class": target_class, "model": model, "noise_path": noise_path}
    df.to_csv(MANIFEST_PATH, index=False)

# Remove entry in the index of the dataset
def remove_entry(index: int) -> None:
    if (df.empty): load_dataset()
    df.drop([index], inplace = True)
    df.to_csv(MANIFEST_PATH, index=False)