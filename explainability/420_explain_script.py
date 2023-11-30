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

dataPrep = pd.get_dummies(dataPrep, drop_first=True)

Y = dataPrep.iloc[:,-1]
X = dataPrep.iloc[:,:-1]


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

from interpret.ext.blackbox import TabularExplainer
 
classes = ['Loan Accepted','Loan Rejected']
features = list(X.columns)

tab_explainer = TabularExplainer(trained_model,
                                 X_train,
                                 features=features,
                                 classes=classes)

# get the global explanations
global_explanation = tab_explainer.explain_global(X_train)

# upload explanation to azure

from azureml.interpret import ExplanationClient

explain_client = ExplanationClient.from_run(new_run)

explain_client.upload_model_explanation(global_explanation,
                                        comment="tabular explanation on Loan applications")



downloaded_explanations = explain_client.download_model_explanation()

feature_importances = downloaded_explanations.get_feature_importance_dict()



new_run.complete()

