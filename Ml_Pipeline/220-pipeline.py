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
myenv_dep = CondaDependencies.create(conda_packages=['scikit-learn','pandas'])
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

# define Pipeline steps
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData

input_ds = ws.datasets.get('loan price applcations using sdk')

datafolder = PipelineData('datafolder',datastore=ws.get_default_datastore())


# step-1 Data preparation step
dataPrep_step = PythonScriptStep(
            name = "01 Data Preparation",
            source_directory="/Users/soumyajitpal/codes/Azure-ml/Ml_Pipeline",
            script_name="220-dataprep_pipeline.py",
            inputs=[input_ds.as_named_input('raw_data')],
            outputs=[datafolder],
            runconfig=run_config,
            arguments=['--datafolder', datafolder])

# step-2 Train the model
train_step = PythonScriptStep(
            name = "02 train the model",
            source_directory="/Users/soumyajitpal/codes/Azure-ml/Ml_Pipeline",
            script_name="220-training_pipeline.py",
            inputs=[datafolder],
            runconfig=run_config,
            arguments=['--datafolder', datafolder])


# configure and build the pipeline
steps = [dataPrep_step, train_step]

from azureml.pipeline.core import Pipeline

new_pipeline = Pipeline(workspace=ws, steps=steps)

# Create the experiment and run the pipeline
new_experiment = Experiment(workspace=ws, name='PipelineExp01')

new_pipeline_run = new_experiment.submit(new_pipeline)

new_pipeline_run.wait_for_completion(show_output=True)

