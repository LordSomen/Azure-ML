from azureml.interpret import ExplanationClient
from azureml.core import Workspace

ws = Workspace.from_config("./config")

new_run = ws.get_run(run_id='Explainer_Exp001_1695023892_8628430e')

explain_client = ExplanationClient.from_run(new_run)

downloaded_explanations = explain_client.download_model_explanation()

feature_importances = downloaded_explanations.get_feature_importance_dict()

print(feature_importances)