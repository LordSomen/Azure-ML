from azureml.core import Workspace, Run




ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")


new_run = Run.get_context()


import pandas as pd

#load the data from the local files
df = pd.read_csv(r"/Users/soumyajitpal/codes/Azure-ml/loan.csv")

# Select columns from the dataset
LoanPrep = df[["Married", 
             "Education",
             "Self_Employed",
             "ApplicantIncome",
             "LoanAmount",
             "Loan_Amount_Term",
             "Credit_History",
             "Loan_Status"
             ]]

# Clean Missing Data - Drop the columns with missing values
LoanPrep = LoanPrep.dropna()


# Create Dummy variables - Not required in designer
LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)

# Create X and Y - Similar to "edit columns" in Train Module
Y = LoanPrep[['Loan_Status_Y']]
X = LoanPrep.drop(['Loan_Status_Y'], axis=1)

# Split Data - X and Y datasets are training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)


# Build the Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# Fit the data to the LogisticRegression object - Train Model
lr.fit(X_train, Y_train)

# Predict the outcome using Test data - Score Model 
# Scored Label
Y_predict = lr.predict(X_test)

# Get the probability score - Scored Probabilities
Y_prob = lr.predict_proba(X_test)[:, 1]

# Get Confusion matrix and the accuracy/score - Evaluate
from sklearn.metrics import confusion_matrix
cm    = confusion_matrix(Y_test, Y_predict)
score = lr.score(X_test, Y_test)


# Log metrics
new_run.log("TotalObservations",len(df))
new_run.log("ConfusedMatrix",cm)
new_run.log("score",score)

# azure style log metrics
cm_dict = {
    "schema_type": "confusion_matrix",
    "schema_version": "v1",
    "data": {
        "class_labels":["N","Y"],
        "matrix": cm.tolist()
    }
}

new_run.log_confusion_matrix("ConfusionMatrixAzure",cm_dict)

# Create the scored dataset and upload the output

X_test = X_test.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)

Y_prob_df = pd.DataFrame(Y_prob, columns=["Scored Probabilities"])
Y_predict_df = pd.DataFrame(Y_predict,columns=["Scored Label"])

scored_dataset = pd.concat([X_test,Y_test, Y_predict_df, Y_prob_df], axis=1)

scored_dataset.to_csv("./outputs/Loan_scored.csv",index=False)

# complete the run
new_run.complete()