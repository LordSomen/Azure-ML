# ----------------------------------------------------------------------
# Consume the webservice using API end points using "requests" library
#
# Steps to use requests module in python
# 1. Get scoring URI
# 2. Prepare the input data and create JSON
# 3. Prepare the headers with authorization key
# 4. Make a POST request to API using URI, headers and data
# 5. Collect the response and deserialize the JSON
# ----------------------------------------------------------------------

import requests
import json

# ------------------------------------------------------
# Set the URI for the web service
# ------------------------------------------------------
scoring_uri = 'http://20.204.203.104:80/api/v1/service/loan-app-service/score'


# ------------------------------------------------------
# Prepare the input data and create the serialized JSON
# ------------------------------------------------------
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

# Convert the input data to a serialized JSON
json_data = json.dumps({"data": x_new})



# ---------------------------------------------------
# Create the headers with authorization key
# ---------------------------------------------------
key = 'RGLN4x2D04KNJttbmk4Vx5AT4HIA4Yei'

# Set the content type and authorization
headers = {'Content-Type': 'application/json',
           'Authorization' : f'Bearer {key}'}


# ---------------------------------------------------
# Make the request using POST method and collect the response
# ---------------------------------------------------
response = requests.post(scoring_uri, json_data, headers=headers)
predicted_classes = json.loads(response.json())

print('\n', predicted_classes)













