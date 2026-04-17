import pandas as pd
import torch
import os

# Path for manfiest.csv file containing the dataset
# The os methods ensure that the script locates manifest.csv correctly 
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.csv")
LOAD_IMAGES = True
# Set LOAD_IMAGES True if we want to preload images

path_df = pd.DataFrame()
images = {}
max_index = 0

# Load dataset and tensor objects
def load():
    global path_df, max_index
    # Load Pandas Dataframe
    path_df = pd.read_csv(MANIFEST_PATH, dtype={"image_path": "str", "target_class": "int32", "model": "str", "noise_path": "str"})
    max_index = path_df.shape[0]

    # Preload tensors
    if (not LOAD_IMAGES): return
    images.clear()
    for index, row in path_df.iterrows():
        try:
            raw_image = torch.load(row["image_path"])
        except:
            print(f'Tried to load the tensor file with path {row["image_path"]}')
        try:
            noise = torch.load(row["noise_path"])
        except:
            print(f'Tried to load the tensor file with path {row["noise_path"]}')
        images[index] = {"image": raw_image, "noise": noise}

# Get the manifest information for an entry
def get_entry(index: int) -> pd.Series:
    assert index in path_df.index
    return path_df.iloc[index]

# Get raw image and noise image from  
def get_images(index: int) -> pd.Series:
    assert index in path_df.index and LOAD_IMAGES
    return images[index]

# Add a new entry, also preloads the images of the entry
def add(image_path: str, target_class: int, model: str, noise_path: str, save_to_csv: bool=False) -> None:
    global images
    if (path_df.empty): load()
    new_index = path_df.index.max()+1
    path_df.loc[new_index] = {"":path_df.shape[0], "image_path": image_path, "target_class": target_class, "model": model, "noise_path": noise_path}
    if (LOAD_IMAGES): images[new_index] = ({"image": torch.load(image_path), "noise": torch.load(noise_path)})
    if (save_to_csv): to_csv()

# Remove an entry, also deletes the preloaded images of the entry
def remove(index: int, save_to_csv: bool=False) -> None:
    if (path_df.empty): load()
    assert index in path_df.index
    path_df.drop([index], inplace = True)
    if (LOAD_IMAGES): images.pop(index)
    if (save_to_csv): to_csv()

# Save dataframe to manifest.csv
def to_csv():
    path_df.to_csv(MANIFEST_PATH, index=False)