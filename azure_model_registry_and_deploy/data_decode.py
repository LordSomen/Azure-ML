import pandas as pd 

data2 = pd.read_csv(r"/Users/soumyajitpal/codes/Azure-ml/loan.csv")


columns = data2.select_dtypes(include='number').columns

import joblib
obj_file = "./outputs/scaler.pkl"

sc = joblib.load(obj_file)

data2[columns] = sc.transform(data2[columns])

print(data2)



