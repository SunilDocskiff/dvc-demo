from datetime import date

import pandas as pd

from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
test_df = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))



train_df.drop(['label'],axis=1).to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_df.drop(['label'],axis=1).to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

train_df.label.to_csv(
    str(Config.FEATURES_PATH / "train_labels.csv"), index=None
)
test_df.label.to_csv(
    str(Config.FEATURES_PATH / "test_labels.csv"), index=None
)