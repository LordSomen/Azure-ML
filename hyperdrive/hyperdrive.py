from azureml.core import Workspace, Experiment

ws = Workspace.from_config(path="./config/")

input_ds = ws.datasets.get('loan price applcations using sdk')




# environment
from azureml.core import Environment
from azureml.core.environment import CondaDependencies

myenv = Environment(name="myenv")

myenv_dep = CondaDependencies.create(conda_packages=['scikit-Learn','pip','pandas'],
                                     pip_packages=['azureml-defaults'])

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


# create a script
from azureml.core import ScriptRunConfig

script_config = ScriptRunConfig(source_directory=r"/Users/soumyajitpal/codes/Azure-ml/hyperdrive",
                                script="320-hyperdrive_script.py",
                                arguments=['--input-data',input_ds.as_named_input('raw_data')],
                                environment=myenv,
                                compute_target=compute_cluster)




from azureml.train.hyperdrive import GridParameterSampling , choice , PrimaryMetricGoal

hyper_params = GridParameterSampling(
        {
            '--n_estimators': choice(10,20,50,100),
            '--min_samples_leaf': choice(1,2,5)
        })

# configure the hyperdrive class

from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal

hyper_config = HyperDriveConfig(run_config=script_config,
                                hyperparameter_sampling=hyper_params,
                                policy=None,
                                primary_metric_name='accuracy',
                                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                max_total_runs=5,
                                max_concurrent_runs=2)


new_experiment = Experiment(workspace=ws, name='hyperdrive-Exp001')
new_run = new_experiment.submit(config=hyper_config)

new_run.wait_for_completion(show_output=True)


best_run = new_run.get_best_run_by_primary_metric()

print("Best Run ID: ", best_run.id)
print(best_run.get_metrics())







