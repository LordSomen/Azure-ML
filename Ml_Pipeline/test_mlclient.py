from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential , EnvironmentCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# authenticate
credential = DefaultAzureCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="put your subscription id",
    resource_group_name="AzureML-SP-RG10",
    workspace_name="Azureml-SDK-WS10",
)

v1 = "initial"
data_asset = ml_client.data.get(name="LoanApp", version=v1)
print(data_asset)