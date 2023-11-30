from azureml.core import Run 

new_run = Run.get_context()

ws = new_run.experiment.workspace

def data_prep():
    input_ds = ws.datasets.get('loan price applcations using sdk').to_pandas_dataframe()
    input_ds = input_ds.drop(['Loan_ID'],axis=1)
    return input_ds