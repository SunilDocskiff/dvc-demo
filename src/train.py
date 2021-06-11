import pickle
import yaml

import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from config import Config


params = yaml.safe_load(open("params.yaml"))["train"]

seed = params["seed"]
n_est = params["n_est"]
min_split = params["min_split"]
max_depth=params["max_depth"]

Config.MODELS_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(str(Config.FEATURES_PATH / "train_features.csv"))
y_train = pd.read_csv(str(Config.FEATURES_PATH / "train_labels.csv"))

#model = OneVsRestClassifier(LinearSVC(random_state=42))
base_model=RandomForestClassifier(n_estimators=n_est, min_samples_split=min_split, n_jobs=2, max_depth=params["max_depth"],random_state=seed)
model = OneVsRestClassifier(base_model)
model = model.fit(X_train, y_train.to_numpy().ravel())



pickle.dump(model, open(str(Config.MODELS_PATH / "model.pickle"), "wb"))