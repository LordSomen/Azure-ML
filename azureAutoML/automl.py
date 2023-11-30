from azureml.core import Workspace, Run


ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

input_ds = ws.datasets.get('loan price applcations using sdk')

cluster_name = 'my-cluster-001'

from azureml.core.compute import AmlCompute

print("Accesing the compute cluster")

if cluster_name not in ws.compute_targets:
    print("Creating the compute cluster with name",cluster_name)
    compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_D11_V2',
            max_nodes=2)
    
    cluster = AmlCompute.create(ws, cluster_name, compute_config)
    cluster.wait_for_completion()
else:
    cluster = ws.compute_targets[cluster_name]
    print(cluster_name," , compute cluster found. Using it...")




from azureml.train.automl import AutoMLConfig

print("Creating the AutoML Configuration.....")

automl_config = AutoMLConfig(task='classification',
                             compute_target=cluster,
                             training_data=input_ds,
                             validation_size=0.3,
                             label_column_name="Loan_Status",
                             primary_metric="norm_macro_recall",
                             iterations=2,
                             max_concurrent_iterations=2,
                             experiment_timeout_hours=0.5,
                             featurization='auto')


from azureml.core.experiment import Experiment

new_exp = Experiment(ws , 'azureml-sdk-automl-exp-02')

print("Submitting the experiment......")

new_run = new_exp.submit(automl_config)
new_run.wait_for_completion(show_output=True)

