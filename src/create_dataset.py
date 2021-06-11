import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from config import Config

np.random.seed(Config.RANDOM_SEED)


params = yaml.safe_load(open("params.yaml"))["prepare"]

Config.ORIGINAL_DATASET_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)

gdown.download(
    "https://drive.google.com/uc?id=1sWbd-uAz9dYq3uH6aSWmhwWhrFlwCcnG",
    str(Config.ORIGINAL_DATASET_FILE_PATH),
)

df = pd.read_csv(str(Config.ORIGINAL_DATASET_FILE_PATH))

df_train, df_test = train_test_split(
    df, test_size=params["split"], random_state=params["seed"],
)

df_train.to_csv(str(Config.DATASET_PATH / "train.csv"), index=None)
df_test.to_csv(str(Config.DATASET_PATH / "test.csv"), index=None)