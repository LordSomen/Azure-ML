from azureml.core import Workspace, Experiment

print("Accessing the workspace....")
ws = Workspace.from_config("/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

print("Accessing the loan dataset......")
input_ds = ws.datasets.get('loan price applcations using sdk')


# from azureml.core import Environment
# from azureml.core.environment import CondaDependencies

# myenv2 = Environment(name="myenv2")

# myenv_dep = CondaDependencies.create(conda_packages=['scikit-Learn==1.3.0','pip','pandas'],
#                                      pip_packages=['azureml-defaults'],
#                                      python_version='3.9.0')

# myenv2.python.conda_dependencies = myenv_dep

# # Register the env
# myenv2.register(ws)

myenv = ws.environments['myenv']

# specify the cluster name

cluster_name = "pipeline-ack02"

if cluster_name not in ws.compute_targets:
    print("crating the prodcution test cluster.....")
    from azureml.core.compute import AksCompute

    print("provisoning the cluster")
    aks_config = AksCompute.provisioning_configuration(location='centralindia',
                                                    vm_size='STANDARD_D12_V2',
                                                    agent_count=1,
                                                    cluster_purpose='DevTest')

    from azureml.core.compute import ComputeTarget

    production_test_cluster = ComputeTarget.create(ws,cluster_name,aks_config)
    production_test_cluster.wait_for_completion(show_output=True)
else:
    print(cluster_name," ...cluster exists.")
    production_test_cluster = ws.compute_targets[cluster_name]



###### create the inference configuration
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(
    source_directory='/Users/soumyajitpal/codes/Azure-ml/azure_model_registry_and_deploy/service_files',
    environment=myenv,
    entry_script='scoring_script.py')

# deployment

from azureml.core.webservice import AksWebservice

deploy_config = AksWebservice.deploy_configuration(cpu_cores=1,
                                                   memory_gb=1,
                                                   autoscale_enabled=False,
                                                   num_replicas=1)


# Deploy the webservice

from azureml.core.model import Model

model = ws.models["LoanApproval_model"]

service = Model.deploy(workspace=ws,
                       name='loan-app-service',
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deploy_config,
                       deployment_target=production_test_cluster)


service.wait_for_deployment(show_output=True)


