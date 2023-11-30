import pandas as pd 

df = pd.read_csv(r"/Users/soumyajitpal/codes/Azure-ml/loan.csv")

data1 = df.copy()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

columns = data1.select_dtypes(include="number").columns

scaler_fitted = scaler.fit(data1[columns])

data1[columns] = scaler_fitted.transform(data1[columns])

# create the seralised object
import joblib

obj_file = "./outputs/scaler.pkl"

joblib.dump(value=scaler_fitted, filename=obj_file)