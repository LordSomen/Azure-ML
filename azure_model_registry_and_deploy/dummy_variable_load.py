import joblib
import pandas as pd

obj_file = './outputs/columns.pkl'

ref_cols = joblib.load(obj_file)

X_deploy = pd.DataFrame(
    {
        'name':['Jitesh'],
        'income':[70],
        'education':['Post-Grad1']
    }
)

X_deploy_enc = pd.get_dummies(X_deploy)

deploy_cols = X_deploy_enc.columns

missing_cols = ref_cols.different(deploy_cols)

for cols in missing_cols:
    X_deploy_enc[cols] = 0

X_deploy_enc = X_deploy_enc[ref_cols]

extra_cols = deploy_cols.difference(ref_cols)

if len(extra_cols):
    print("there is an unknown category")

