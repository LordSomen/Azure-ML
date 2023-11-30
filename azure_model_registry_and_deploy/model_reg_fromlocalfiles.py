from azureml.core import Workspace, Experiment, Model

print("Accessing the workspace....")
ws = Workspace.from_config(r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

Model.register(
    workspace = ws,
    model_path='./outputs/models.pkl',
    model_name='LoanApproval_model_local',
    tags={'source':'SDK-local','algorithm':'RandomForest'},
    properties={'Accuracy':0.786},
    description='Combined Models from the Run.'
)