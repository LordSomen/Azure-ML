from azureml.core import Workspace, Experiment

print("Accessing the workspace....")
ws = Workspace.from_config(r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

print("Accessing the loan dataset......")
input_ds = ws.datasets.get('loan price applcations using sdk')

# ---------------------------------------------------------------------
# Create the experiment and run
# ---------------------------------------------------------------------
print("Creating the experiment...")
experiment = Experiment(workspace=ws, name='Webservice_Explainer_Exp001')


# run experiment
print("start logging.......")
new_run = experiment.start_logging()

import pandas as pd

df = input_ds.to_pandas_dataframe()

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

print(list(experiment.get_runs()))





