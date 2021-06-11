import json
import pickle

import pandas as pd
from sklearn.metrics import f1_score
from config import Config

X_test = pd.read_csv(str(Config.FEATURES_PATH / "test_features.csv"))
y_test = pd.read_csv(str(Config.FEATURES_PATH / "test_labels.csv"))

model = pickle.load(open(str(Config.MODELS_PATH / "model.pickle"), "rb"))

score = model.score(X_test, y_test)

y_pred = model.predict(X_test)
f1_score = f1_score(y_test, y_pred, average='macro')
#print(model.estimator.__class__.__name__)
with open(str(Config.METRICS_FILE_PATH), "w") as outfile:
    json.dump(dict(model=model.estimator.__class__.__name__ , f1_score=f1_score), outfile)