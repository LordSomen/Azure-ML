from azureml.core import Workspace, Experiment, ScriptRunConfig

# Access the workspace

ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")


# specify the cluster name

cluster_name = "my-cluster-sdk001"

#provisoning configuration using AmlCompute
from azureml.core.compute import AmlCompute

# configuration of the compute cluster
compute_config = AmlCompute.provisioning_configuration(
                        vm_size="STANDARD_D11_V2",
                        max_nodes=2
                    )
from azureml.core.compute import ComputeTarget
# create the cluster
cluster = ComputeTarget.create(ws, cluster_name, compute_config)
