import sys
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


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

print("ref:", ref["cell gap(um)"])
print("Tr:", ref["Tr(ms)"][0])
ref_Tr = ref["Tr(ms)"][0]
ref_cell_gap = ref["cell gap(um)"][0]
ref_LC = ref["LC"][0]

rt_cell_gap = pd.merge(rt, axo[["ID", "cell gap", "point"]], how="left", on=["ID", "point"])
df = rt_cell_gap[rt_cell_gap["LC"] == ref_LC].copy()
df["Tr"] = df["Rise-mean (10-90)"]
df["Vop"] = df["Target Vpk"]

df = df.groupby(by=["ID", "Vop", "point"], as_index=False).mean()

# sns.scatterplot(data=df, x="Vop", y="Tr")

model = {}
model["scalar"] = {}
# Let's try some fasion ML (XD
training_set, test_set = train_test_split(
    df,
    test_size = 0.3,
    random_state = 42
)
X_train = training_set[["Tr", "cell gap"]].to_numpy()
y_train = training_set["Vop"].to_numpy()
X_test = test_set[["Tr", "cell gap"]].to_numpy()
y_test = test_set["Vop"].to_numpy()
model["scalar"]["ref"] = StandardScaler().fit(X_train)
X_train = model["scalar"]["ref"].transform(X_train)
X_test = model["scalar"]["ref"].transform(X_test)
valid_data = model["scalar"]["ref"].transform([[ref_Tr, ref_cell_gap]])
# eXtreme Grandient Boostng Regression
model["Vop_ref_XGBR"] = XGBRegressor(
    n_estimators = 50,
    learning_rate = 0.1,
    max_depth = 3,
    gamma = 0.01,
    reg_lambda = 0.01
)
model["Vop_ref_XGBR"].fit(
    X_train, y_train,
    early_stopping_rounds = 10,
    eval_set = [(X_test, y_test)],
    verbose = False
)
print("R2_train:", model["Vop_ref_XGBR"].score(X_train, y_train))
print("R2_test:", model["Vop_ref_XGBR"].score(X_test, y_test))
ref_Vop = float(model["Vop_ref_XGBR"].predict(valid_data))
print("Vop from Ref[Tr, cell gap]:", ref_Vop)

# Calculate RT, Tf, Tr
df = rt_cell_gap.copy()
df["Vop"] = df["Target Vpk"]
df["RT"] = df["Rise-mean (10-90)"] + df["Fall-mean (10-90)"]
df["Tr"] = df["Rise-mean (10-90)"]
df["Tf"] = df["Fall-mean (10-90)"]
training_set, test_set = train_test_split(
    df,
    test_size = 0.1,
)

model["rt"] = {}
Tr = {}
Tf = {}
RT = {}

for LC in cond["LC"].unique():
    print(LC)
    model["rt"][LC] = {}
    X_train = training_set[training_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    X_test = test_set[test_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    # Standarize & Normalize
    model["scalar"][f'rt_{LC}'] = StandardScaler().fit(X_train)
    X_train = model["scalar"][f'rt_{LC}'].transform(X_train)
    X_test = model["scalar"][f'rt_{LC}'].transform(X_test)
    valid_data = model["scalar"][f'rt_{LC}'].transform([[ref_Vop, ref_cell_gap]])
    for item in ["Tr", "Tf", "RT"]:
        y_train = training_set[training_set["LC"]==LC][item].to_numpy()
        y_test = test_set[test_set["LC"]==LC][item].to_numpy()
        
        model["rt"][LC][f"{item}_XGBR"] = XGBRegressor(
            n_estimators = 3,
            learning_rate = 1,
            max_depth = 3,
            gamma = 1,
            reg_lambda = 1
        ).fit(
            X_train, y_train,
            early_stopping_rounds = 10,
            eval_set = [(X_test, y_test)],
            verbose = False, 
        )
        print(f'RT test {model["rt"][LC][f"{item}_XGBR"].score(X_test, y_test):.2f}')
        ans = float(model["rt"][LC][f"{item}_XGBR"].predict(valid_data))
        print(f"{LC}: {item}: {ans:.2f} ms")
opt_cell_gap = pd.merge(
    opt,
    axo[["ID", "point", "cell gap"]], 
    left_on=["ID", "Point"], 
    right_on=["ID", "point"], 
    how="left"
)
model["opt"] = {}
df = opt_cell_gap.copy()
df["T%"] = opt_cell_gap.groupby(by=["ID", "Point"])["LCM_Y%"].apply(lambda x: 100*x / float(x.max()))
df["Vop"] = df["Voltage"]/2.0
df["LC%"] = df["LCM_Y%"]
df = df[df["Vop"] > 2]
training_set, test_set = train_test_split(
    df,
    test_size = 0.2,
#     random_state = 42
)
LCp = {}
T = {}
Wx = {}
Wy = {}
for LC in cond["LC"].unique():
    print(LC)
    model["opt"][LC] = {}
    X_train = training_set[training_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    y_train = training_set[training_set["LC"]==LC]["T%"].to_numpy()
    X_test = test_set[test_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    y_test = test_set[test_set["LC"]==LC]["T%"].to_numpy()
    # Standarize & Normalize
    model["scalar"][f'opt_{LC}'] = StandardScaler().fit(X_train)
    X_train = model["scalar"][f'opt_{LC}'].transform(X_train)
    X_test = model["scalar"][f'opt_{LC}'].transform(X_test)
    valid_data = model["scalar"][f'opt_{LC}'].transform([[ref_Vop, ref_cell_gap]])
    # eXtreme Grandient Boostng Regression
    model["opt"][LC]["T_XGBR"] = XGBRegressor(
        n_estimators = 3,
        learning_rate = 1,
        max_depth = 3,
        gamma = 1,
        reg_lambda = 1
    )
    model["opt"][LC]["T_XGBR"].fit(
        X_train, y_train,
        early_stopping_rounds = 10,
        eval_set = [(X_test, y_test)],
        verbose = False, 
    )
#     print("R2_train:", f["opt"][LC]["T_XGBR"].score(X_train, y_train))
    print("R2_test:", model["opt"][LC]["T_XGBR"].score(X_test, y_test))
    T[LC] = float(model["opt"][LC]["T_XGBR"].predict(valid_data))
    print("T%", LC ,"", T[LC])
    y_train = training_set[training_set["LC"]==LC]["W_x"].to_numpy()
    y_test = test_set[test_set["LC"]==LC]["W_x"].to_numpy()
    model["opt"][LC]["Wx_XGBR"] = XGBRegressor(
        n_estimators = 1000,
        learning_rate = 0.01,
        max_depth = 7,
        gamma = 0.001,
        reg_lambda = 0.1,
    )
    model["opt"][LC]["Wx_XGBR"].fit(
        X_train, y_train,
        early_stopping_rounds = 10,
        eval_set = [(X_test, y_test)],
        verbose = False, 
    )
#     print("R2_train:", model["opt"][LC]["Wx_XGBR"].score(X_train, y_train))
    print("R2_test:", model["opt"][LC]["Wx_XGBR"].score(X_test, y_test))
    Wx[LC] = float(model["opt"][LC]["Wx_XGBR"].predict(valid_data))
    print("Wx", LC ,"", Wx[LC])
    
    y_train = training_set[training_set["LC"]==LC]["W_y"].to_numpy()
    y_test = test_set[test_set["LC"]==LC]["W_y"].to_numpy()
    model["opt"][LC]["Wy_XGBR"] = XGBRegressor(
        n_estimators = 100,
        learning_rate = 1,
        max_depth = 5,
        gamma = 0.0001,
        reg_lambda = 0,
    )
    model["opt"][LC]["Wy_XGBR"].fit(
        X_train, y_train,
        early_stopping_rounds = 10,
        eval_set = [(X_test, y_test)],
        verbose = False, 
    )
#     print("R2_train:", model["opt"][LC]["Wy_XGBR"].score(X_train, y_train))
    print("R2_test:", model["opt"][LC]["Wy_XGBR"].score(X_test, y_test))
    Wy[LC] = float(model["opt"][LC]["Wy_XGBR"].predict(valid_data))
    print("Wy", LC ,"", Wy[LC])
    
    y_train = training_set[training_set["LC"]==LC]["LC%"].to_numpy()
    y_test = test_set[test_set["LC"]==LC]["LC%"].to_numpy()
    # eXtreme Grandient Boostng Regression
    model["opt"][LC]["LC_XGBR"] = XGBRegressor(
        n_estimators = 3,
        learning_rate = 1,
        max_depth = 3,
        gamma = 1,
        reg_lambda = 1
    )
    model["opt"][LC]["LC_XGBR"].fit(
        X_train, y_train,
        early_stopping_rounds = 10,
        eval_set = [(X_test, y_test)],
        verbose = False, 
    )
    print("R2_test:", model["opt"][LC]["LC_XGBR"].score(X_test, y_test))
    LCp[LC] = float(model["opt"][LC]["LC_XGBR"].predict(valid_data))
    print("LC%", LC ,"",LCp[LC])   
    print()

# Generate table
summary_table = pd.DataFrame(
    columns=["LC", "platform", "V90", "V95", "V99", "Vmax", "Vop(V)", "Vop_T%", "Δnd(nm)", "Gap(um)", "LC%", "Wx", "Wx_gain", "Wy", "Wy_gain", "u'", "v'", "Ea", "Eb", "ΔEab", "CR", "ΔCR", "T%", "Scatter", "D", "W", "Tr(ms)", "Tf(ms)", "RT(ms)", "G2G(ms)"]
)
# cell gap range
# +- 0.5 um
cell_gap_range = np.linspace(ref_cell_gap-0.5, ref_cell_gap+0.5, 11)
for LC in cond["LC"].unique():
    for cell_gap in cell_gap_range:
        summary_table = summary_table.append({"LC": LC, "Gap(um)": cell_gap}, ignore_index=True)
        # rt
        X = [[ref_Vop, cell_gap]]
        X_scalar = model["scalar"][f'rt_{LC}'].transform(X)
        Tr = model["rt"][LC]["Tr_XGBR"].predict(X_scalar)
        Tf = model["rt"][LC]["Tf_XGBR"].predict(X_scalar)
        RT = model["rt"][LC]["RT_XGBR"].predict(X_scalar)
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "RT(ms)"] = RT[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Tr(ms)"] = Tr[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Tf(ms)"] = Tf[0]
        # opt
        X_scalar = model["scalar"][f'opt_{LC}'].transform(X)
        Wx = model["opt"][LC]["Wx_XGBR"].predict(X_scalar)
        Wy = model["opt"][LC]["Wy_XGBR"].predict(X_scalar)
        T = model["opt"][LC]["T_XGBR"].predict(X_scalar)
        LCp = model["opt"][LC]["LC_XGBR"].predict(X_scalar)
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Wx"] = Wx[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Wy"] = Wy[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "T%"] = T[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "LC%"] = LCp[0] * 100

summary_table.to_sql("summary", con=engine, if_exists="append", index=False)