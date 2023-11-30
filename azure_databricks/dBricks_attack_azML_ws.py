from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.compute import DatabricksCompute, ComputeTarget
# Access the workspace

ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

print("Initializing Parameters............")
db_resource_group = "Databricks_RG_001"
db_workspace_name = "Databricks_WS_001"
db_access_token = "put your access token"
db_compute_name = "mbdbcluster001"


if db_compute_name not in ws.compute_targets:
    print("creating the configuration")
    attach_config = DatabricksCompute.attach_configuration(
            resource_group= db_resource_group,
            workspace_name= db_workspace_name,
            access_token= db_access_token)
    
    print("attching the compute target")
    db_cluster = ComputeTarget.attach(ws,
                                      db_compute_name,
                                      attach_config)
    db_cluster.wait_for_completion(True)
else:
    print("already attached......")
    db_cluster = ws.compute_targets[db_compute_name]


    