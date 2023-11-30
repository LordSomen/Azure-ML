from azureml.core import Workspace, Experiment

print("Accessing the workspace....")
ws = Workspace.from_config("./config")

print("Accessing the loan dataset......")
input_ds = ws.datasets.get('loan price applcations using sdk')


from azureml.core import Environment
from azureml.core.environment import CondaDependencies

myenv = Environment(name="myenv")

myenv_dep = CondaDependencies.create(conda_packages=['scikit-Learn','pip','pandas'],
                                     pip_packages=['azureml-defaults','azureml-interpret'])

myenv.python.conda_dependencies = myenv_dep

# Register the env
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


# ---------------------------------------------------------------------
# Create a script configuration for custom environment of myenv
# ---------------------------------------------------------------------
from azureml.core import ScriptRunConfig

print("Creating the ScriptRunConfig....")
script_config = ScriptRunConfig(source_directory=".",
                                script="420_explain_script.py",
                                arguments = ['--input-data', input_ds.as_named_input('raw_data')],
                                environment=myenv,
                                compute_target=compute_cluster)


# ---------------------------------------------------------------------
# Create the experiment and run
# ---------------------------------------------------------------------
print("Creating the experiment...")
new_experiment = Experiment(workspace=ws, name='Explainer_Exp001')

print("Submittting the experiment...")
new_run = new_experiment.submit(config=script_config)

new_run.wait_for_completion(show_output=True)



