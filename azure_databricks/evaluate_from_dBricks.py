from azureml.core import Run
import argparse

new_run = Run.get_context()

ws = new_run.experiment.workspace
print("workspace accessed........")


parser = argparse.ArgumentParser()
parser.add_argument("--testdata", type=str)
args = parser.parse_args()
print("arguments accessed and printing.....")
print(args.testdata)

############## do your code below this
### access the data from databricks

import os
import pandas as pd

path = os.path.join(args.testdata, 'x_test.csv')
X_test = pd.read_csv(path)

path = os.path.join(args.testdata,"y_test.csv")
Y_test = pd.read_csv(path)

path = os.path.join(args.testdata,"y_predict.csv")
Y_predict = pd.read_csv(path)

import joblib
objfile = os.path.join(args.testdata,'rfcModel.pkl')
rfc = joblib.load(objfile)


#================ evaluation===============================
Y_prob = rfc.predict_proba(X_test)[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = rfc.score(X_test,Y_test)
print(score)

cm_dict = {"schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {"class_labels": ["N", "Y"],
                    "matrix": cm.tolist()}
           }

new_run.log_confusion_matrix("ConfusionMatrix", cm_dict)
new_run.log("Score", score)

# complete the run
new_run.complete()
