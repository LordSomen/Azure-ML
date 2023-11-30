from azureml.core import Workspace, Experiment, ScriptRunConfig

# Access the workspace

ws = Workspace.from_config(path=r"/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

# using your environment
from azureml.core import Environment

myenv = Environment("user-managed-env")
myenv.python.user_managed_dependencies = True

# You can choose a specific Python environment by pointing to a Python path 
# myenv.python.interpreter_path = '/home/johndoe/miniconda3/envs/myenv/bin/python'

# new experiment

new_experiment = Experiment(workspace=ws,
                            name="loan_Script")

script_config = ScriptRunConfig(source_directory=r"/Users/soumyajitpal/codes/Azure-ml/Ml_Pipeline",
                                script="180-script_to_run.py",
                                compute_target='local',
                                environment=myenv)

new_run = new_experiment.submit(config=script_config)

# create a wait for completion
new_run.wait_for_completion()