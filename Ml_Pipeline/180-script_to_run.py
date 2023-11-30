from azureml.core import Workspace, Datastore, Dataset , Experiment
from azureml.core import Run

ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

az_store = Datastore.get(ws,"azure_sdk_blob1")
az_dataset = Dataset.get_by_name(ws,name="loan price applcations using sdk")
az_default_store = ws.get_default_datastore()

##=============================
# ==== Use the run from the Job currently running======
# ================================ 

new_run = Run.get_context()

########### Processing Code ###################
# convert data asset to pandas dataframe
df = az_dataset.to_pandas_dataframe()

# count the observations
total_observations = len(df)

# get the null/missing values
null_df = df.isnull().sum()


# create a new dataframe with new features
new_df = df[['Gender','Married','Education','Loan_Status']]
new_df.to_csv("./outputs/loan_trunc.csv",index=False)

## log the metrics to the workspace
for column in df.columns:
    new_run.log(column,null_df[column])

#### complete running of an experiment
new_run.complete()