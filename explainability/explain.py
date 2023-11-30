import pandas as pd
import pprint

df = pd.read_csv("/Users/soumyajitpal/codes/Azure-ml/loan.csv")

dataPrep = df.drop(['Loan_ID'],axis=1)

dataPrep = dataPrep.dropna()

dataPrep = pd.get_dummies(dataPrep, drop_first=True)
print(dataPrep)
Y = dataPrep[['Loan_Status_Y']]
X = dataPrep.drop(['Loan_Status_Y'], axis=1)


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

# ===================================================================
# explnations create

from interpret.ext.blackbox import TabularExplainer
 
classes = ['Loan Accepted','Loan Rejected']
features = list(X.columns)

tab_explainer = TabularExplainer(trained_model,
                                 X_train,
                                 features=features,
                                 classes=classes)

# get the global explanations
global_explanation = tab_explainer.explain_global(X_train)

# get the feature importance data
global_fi = global_explanation.get_feature_importance_dict()
print("global explanations")
print(global_fi)

X_explain = X_test[0:10]

print(X_explain)

local_explanation = tab_explainer.explain_local(X_explain)

local_features = local_explanation.get_ranked_local_names()
local_importance = local_explanation.get_ranked_local_values()

print(local_features)
print(local_importance)

for i in range(0,len(local_features)):
    labels = local_features[i]
    print("\n Feature support values for :", classes[i])

    for j in range(0, len(labels)):

        if Y_predict[j] == i:
            print("\n\t Observation number : ",j+1)
            feature_names = labels[j]

            print("\t\t", "Feature Name".ljust(30),"  Value")
            print("\t\t","-"*30,"-"*10)

            for k in range(0,len(feature_names)):
                print("\t\t", feature_names[k].ljust(30), round(local_importance[i][j][k],6))