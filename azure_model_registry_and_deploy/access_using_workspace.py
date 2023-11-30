# -------------------------------------------------------------
# Consume the service end point using workspace access.
# -------------------------------------------------------------

# Import the Azure ML classes
from azureml.core import Workspace

# Access the workspace using config file
print("Accessing the workspace....")
ws = Workspace.from_config("/Users/soumyajitpal/codes/Azure-ml/DataConnections&Handling/config")

# Access the service end points
print("Accessing the service end-points")
service = ws.webservices['loan-app-service']


# Prepare the input data
import json

x_new = {
        'Gender': 'Female',
        'Married': 'No',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5849,
        'CoapplicantIncome': 0.0,
        'LoanAmount': 264,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }

# Convert the dictionary to a serializable list in json
json_data = json.dumps({"data": x_new})
print(json_data)
# Call the web service
print("Calling the service...")
try:
    response = service.run(input_data = json_data)

    # Collect and convert the response in local variable
    print("Printing the predicted class...")
    predicted_classes = json.loads(response)

    print('\n', predicted_classes)
except Exception as e:
    print(e)
    




   
    
    