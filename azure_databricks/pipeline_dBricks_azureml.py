from azureml.core import Workspace, Experiment, ScriptRunConfig

# Access the workspace

ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

# custom environment
# using your environment
# from azureml.core import Environment

# myenv = Environment("user-managed-env")
# myenv.python.user_managed_dependencies = True

# -------------------------------------------------
# Create custom environment
from azureml.core import Environment
from azureml.core.environment import CondaDependencies

# Create the environment
myenv = Environment(name="MyEnvironment")

# Create the dependencies object
myenv_dep = CondaDependencies.create(conda_packages=['scikit-learn','pip','pandas','joblib'],
                                     pip_packages=['azureml-defaults'])
myenv.python.conda_dependencies = myenv_dep

# Register the environment
myenv.register(ws)


# specify the cluster name

cluster_name = "pipeline-cluster-sdk001"

#provisoning configuration using AmlCompute
from azureml.core.compute import AmlCompute

if cluster_name not in ws.compute_targets:

    # configuration of the compute cluster
    compute_config = AmlCompute.provisioning_configuration(
                            vm_size="STANDARD_D11_V2",
                            max_nodes=2
                        )
    from azureml.core.compute import ComputeTarget
    # create the cluster
    compute_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_cluster.wait_for_completion()
else:
    compute_cluster = ws.compute_targets[cluster_name]
    print(cluster_name," , compute cluster found. Using it...")

#--------------------------------------------------------------
from azureml.core.runconfig import RunConfiguration
run_config = RunConfiguration()

run_config.target = compute_cluster
run_config.environment = myenv

#=============================================================
from azureml.core.compute import DatabricksCompute

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


#================================================================
# Create Data reference of Input and Output
#=================================================================

from azureml.data.data_reference import DataReference
from azureml.pipeline.core import PipelineData

data_store = ws.datastores.get('azure_sdk_blob1')

input_data = DataReference(datastore= data_store,
                           data_reference_name="input")

output_data1 = PipelineData('testdata',datastore=data_store)



# Create the Databricks Step
from azureml.pipeline.steps import DatabricksStep
from azureml.core.databricks import PyPiLibrary

scikit_learn_db = PyPiLibrary(package="scikit-learn")
joblib_db = PyPiLibrary(package='joblib')


notebook_path = r"/Users/soumyajit637@gmail.com/demo_pipeline001"

db_step01 = DatabricksStep(name = "db_step01",
                           inputs= [input_data],
                           outputs= [output_data1],
                           num_workers= 1,
                           notebook_path= notebook_path,
                           run_name="db_notebook_demo",
                           compute_target= db_cluster,
                           pypi_libraries= [scikit_learn_db,joblib_db],
                           allow_reuse= False)


#===============================================
##### create the pipeline step to run python script
#=====================================================
from azureml.pipeline.steps import PythonScriptStep

eval_step = PythonScriptStep(name="Evaluate",
                             source_directory=".",
                             script_name="evaluate_from_dBricks.py",
                             inputs=[output_data1],
                             runconfig=run_config,
                             arguments=['--testdata',output_data1])


#==============Build and submit the pipeline=======================

from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

steps = [db_step01, eval_step]
new_pipeline = Pipeline(workspace=ws,steps=steps)
new_pipeline_run = Experiment(ws,"DB_Notebook_exp001").submit(new_pipeline)

# wait for completion
new_pipeline_run.wait_for_completion(show_output=True)






