from azureml.core import Workspace, Experiment

print("Accessing the workspace....")
ws = Workspace.from_config(r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

new_run = ws.get_run('put your run id')

new_run.register_model(
    model_path='/outputs/models.pkl',
    model_name='LoanApproval_model',
    tags={'source':'SDK-Run','algorithm':'RandomForest'},
    properties={'Accuracy':new_run.get_metrics()['accuracy']},
    description='Combined Models from the Run.'
)