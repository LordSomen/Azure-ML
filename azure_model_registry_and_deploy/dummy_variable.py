import pandas as pd

X_train = pd.DataFrame({
    'name':['Jitesh','Rahul','John','Bill'],
    'income':[40,50,60,70],
    'education':['Grad','Grad','Post-Grad','Post-Grad']
})

X_train_enc = pd.get_dummies(X_train)

train_enc_cols = X_train_enc.columns


import joblib

obj_file = './outputs/columns.pkl'

joblib.dump(value=train_enc_cols,filename=obj_file)


# production process
