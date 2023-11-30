from azureml.core import Workspace


# create the Workspace
#---------------------------------------------
ws = Workspace.create(name='Azureml-SDK-WS11',
                     subscription_id="put your subscription id",
                     resource_group='AzureML-SP-RG11',
                     create_resource_group=True,
                     location='centralindia'
                )

#------------------------------------------------
# write configuration to a local config.json file
#================================================
ws.write_config(path='./config')
