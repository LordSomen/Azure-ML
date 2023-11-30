from azureml.core import Run 

new_run = Run.get_context()

ws = new_run.experiment.workspace

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input-data",type=str)

args = parser.parse_args()

import pandas as pd

df = new_run.input_datasets['raw_data'].to_pandas_dataframe()


dataPrep = df.drop(['Loan_ID'],axis=1)

dataPrep = dataPrep.dropna()

Y = dataPrep.iloc[:,-1]
X = dataPrep.iloc[:,:-1]

# creating the dummies separately for X and Y
X = pd.get_dummies(X)

train_enc_cols = X.columns

# for target variable
Y = pd.get_dummies(Y)
print("target variable check",Y)
Y = Y.iloc[:,-1]
print("target variable check",Y)

#==========================================================

from sklearn.model_selection import train_test_split

X_train , X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state= 123, stratify=Y
)

from sklearn.ensemble import RandomForestClassifier as RFC
rfc = RFC(random_state=123)

trained_model = rfc.fit(X_train,Y_train)


Y_predict = rfc.predict(X_test)
Y_prob = rfc.predict_proba(X_test)[:,1]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_predict)
score = rfc.score(X_test,Y_test)

new_run.log("accuracy",score)

#================================================
# save all transformation and models

import joblib
model_file = './outputs/models.pkl'

joblib.dump(value=[train_enc_cols,trained_model],
            filename=model_file)

new_run.complete()










