import json
import pickle

import pandas as pd
from sklearn.metrics import f1_score
from config import Config
import yaml


params = yaml.safe_load(open("params.yaml"))["train"]

seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]
max_depth=params["max_depth"]

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

model = pickle.load(open(str(Config.MODELS_PATH / "model.pickle"), "rb"))

score = model.score(X_test, y_test)

y_pred = model.predict(X_test)
f1_score = f1_score(y_test, y_pred, average='macro')
#print(model.estimator.__class__.__name__)
with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
    json.dump(dict(model="RF", f1_score=f1_score,seed=seed,n_estimators=n_est,min_split=min_split,max_depth=params["max_depth"] ), outfile)

