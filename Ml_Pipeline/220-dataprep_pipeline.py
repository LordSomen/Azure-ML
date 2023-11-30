from azureml.core import Run

new_run = Run.get_context()
ws = new_run.experiment.workspace

import pandas as pd

# Load the data from the data assets of the datastore
df = new_run.input_datasets['raw_data'].to_pandas_dataframe()



# Select columns from the dataset
df= df[["Married", 
             "Education",
             "Self_Employed",
             "ApplicantIncome",
             "LoanAmount",
             "Loan_Amount_Term",
             "Credit_History",
             "Loan_Status"
             ]]

dataPrep = df

all_cols = dataPrep.columns


# Check the missing values
dataNull = dataPrep.isnull().sum()


# Replace the missing values of string variable with mode
mode = dataPrep.mode().iloc[0]
cols = dataPrep.select_dtypes(include='object').columns

dataPrep[cols] = dataPrep[cols].fillna(mode)


# Replace numerical columns with mean
mean = dataPrep.mean()
dataPrep = dataPrep.fillna(mean)


# Create Dummy variables - Not required in designer/Classic Studio
dataPrep = pd.get_dummies(dataPrep, drop_first=True)


# Normalise the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns = df.select_dtypes(include='number').columns
dataPrep[columns] = scaler.fit_transform(dataPrep[columns])

print(dataPrep.head())

# get the arguments from pipeline job
from argparse import ArgumentParser as AP

parser = AP()
parser.add_argument('--datafolder',type=str)
args = parser.parse_args()


# create the folder if it does not exist
import os
os.makedirs(args.datafolder, exist_ok=True)

# create the path
path = os.path.join(args.datafolder, 'defaults_prep.csv')

# Write the data preparation output as csf file
dataPrep.to_csv(path, index=False)

# log null values
for column in all_cols:
    new_run.log(column, dataNull[column])

# Complete the run
new_run.complete()
