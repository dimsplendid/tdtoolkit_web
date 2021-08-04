import sys
import os
import pandas as pd
import sqlalchemy as sql
from scipy.interpolate import interp1d

batch = sys.argv[1]
if batch == "":
    print("Do not input batch number")
    exit(1)
engine = sql.create_engine('sqlite:///database/test.db', echo=False)

cond = pd.read_sql(f"SELECT * FROM cond WHERE batch == \"{batch}\"", engine)
axo = pd.read_sql(f"SELECT * FROM axo WHERE batch == \"{batch}\"", engine)
rdl = pd.read_sql(f"SELECT * FROM rdl WHERE batch == \"{batch}\"", engine)
opt = pd.read_sql(f"SELECT * FROM opt WHERE batch == \"{batch}\"", engine)
rt = pd.read_sql(f"SELECT * FROM rt WHERE batch == \"{batch}\"", engine)
prop = pd.read_sql(f"SELECT * FROM prop", engine)
ref = pd.read_sql(f"SELECT * FROM ref WHERE batch == \"{batch}\"", engine)

print("ref:", ref)
print("Tr:", ref["Tr(ms)"][0])
ref_Tr = ref["Tr(ms)"][0]
cell_gap = ref["cell gap(um)"][0]
ref_LC = ref["LC"][0]


df = rt_total_table[rt_total_table["LC"] == "LCT-19-580"].copy()
df["Tr"] = df["Rise-mean (10-90)"]
df["Vop"] = df["Target Vpk"]

df = df.groupby(by=["ID", "Vop", "point"], as_index=False).mean()

sns.scatterplot(data=df, x="Vop", y="Tr")

# Let's try some fasion ML (XD
from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(
    df,
    test_size = 0.3,
    random_state = 42
)
X_train = training_set[["Tr", "cell gap (um)"]].to_numpy()
y_train = training_set["Vop"].to_numpy()
X_test = test_set[["Tr", "cell gap (um)"]].to_numpy()
y_test = test_set["Vop"].to_numpy()
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler().fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)
valid_data = scalar.transform([[Tr, cell_gap]])
# eXtreme Grandient Boostng Regression
from xgboost import XGBRegressor
f["Vop_ref_XGBR"] = XGBRegressor(
    n_estimators = 50,
    learning_rate = 0.1,
    max_depth = 3,
    gamma = 0.01,
    reg_lambda = 0.01
)
f["Vop_ref_XGBR"].fit(
    X_train, y_train,
    early_stopping_rounds = 10,
    eval_set = [(X_test, y_test)],
#     verbose = False
)
print("R2_train:", f["Vop_ref_XGBR"].score(X_train, y_train))
print("R2_test:", f["Vop_ref_XGBR"].score(X_test, y_test))
Vop_ref = float(f["Vop_ref_XGBR"].predict(valid_data))
print("Vop from Ref[Tr, cell gap]:", Vop_ref)