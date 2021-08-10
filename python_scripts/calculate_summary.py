import sys
import os
import numpy as np
import pandas as pd
import sqlalchemy as sql
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn import linear_model
# from xgboost import XGBRegressor

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

# RT part

# print("ref:", ref["cell gap(um)"])
# print("Tr:", ref["Tr(ms)"][0])
def custom_f(X):
    features = np.empty(shape=(len(X), 5), dtype=float)
    features[:, 0] = 1
    features[:, 1] = X[:, 0]
    features[:, 2] = X[:, 1]
    features[:, 3] = X[:, 0] * X[:, 1]
    features[:, 4] = X[:, 0] ** 2
    return features
transformer = FunctionTransformer(custom_f)

ref_Tr = ref["Tr(ms)"][0]
ref_cell_gap = ref["cell gap(um)"][0]
ref_LC = ref["LC"][0]

# check is there axo data
if len(axo) != 0:
    rt_cell_gap = pd.merge(rt, axo[["ID", "Point", "cell gap"]], how="left", on=["ID", "Point"])
else:
    rt_cell_gap = pd.merge(rt, rdl[["ID", "cell gap"]], how="left", on="ID")
    
df = rt_cell_gap[rt_cell_gap["LC"] == ref_LC].copy()
df["Tr"] = df["Rise-mean (10-90)"]
df["Vop"] = df["Target Vpk"]

df = df.groupby(by=["ID", "Vop", "Point"], as_index=False).mean()

# sns.scatterplot(data=df, x="Vop", y="Tr")

model = {}
# Let's try some fasion ML (XD
training_set, test_set = train_test_split(
    df,
    test_size = 0.2,
    random_state = 42
)
X_train = training_set[["Tr", "cell gap"]].to_numpy()
y_train = training_set["Vop"].to_numpy()
X_test = test_set[["Tr", "cell gap"]].to_numpy()
y_test = test_set["Vop"].to_numpy()
valid_data = [[ref_Tr, ref_cell_gap]]

# Linear regression

model["Vop_ref_LR"] = Pipeline([
    ('Scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', linear_model.LinearRegression(fit_intercept=False))]
).fit(
    X_train, y_train,
)

# print("R2_train:", model["Vop_ref_LR"].score(X_train, y_train))
# print("R2_test:", model["Vop_ref_LR"].score(X_test, y_test))
ref_Vop = float(model["Vop_ref_LR"].predict(valid_data))
# print("Vop from Ref[Tr, cell gap]:", ref_Vop)

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

for LC in cond["LC"].unique():
    # print(LC)
    model["rt"][LC] = {}
    X_train = training_set[training_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    X_test = test_set[test_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    valid_data = [[ref_Vop, ref_cell_gap]]
    
    for item in ["Tr", "Tf", "RT"]:
        y_train = training_set[training_set["LC"]==LC][item].to_numpy()
        y_test = test_set[test_set["LC"]==LC][item].to_numpy()

        model["rt"][LC][f"{item}_LR"] = Pipeline([
            ('Scalar', StandardScaler()),
#             ('poly', PolynomialFeatures(degree=1)),
            ('Custom_Transformer', transformer),
            ('linear', linear_model.TheilSenRegressor(fit_intercept=False))
        ]).fit(
            X_train, y_train,
        )
        # print(f'R2_test {model["rt"][LC][f"{item}_LR"].score(X_test, y_test):.2f}')
        # ans = float(model["rt"][LC][f"{item}_LR"].predict(valid_data))
        # print(f"{LC}: {item}: {ans:.2f} ms")
        # print()

# OPT Part

# custom transform function for fitting
def opt_features_extract(X):
    # T%(LC%) = a * Vop^4 + b * Vop^3 + c * Vop^2 + d * Vop * cell_gap + e * cell_gap
    #         + f * Vop + g
    features = np.empty(shape=(len(X), 7), dtype=float)
    features[:, 0] = 1
    features[:, 1] = X[:, 0]
    features[:, 2] = X[:, 1]
    features[:, 3] = X[:, 0] * X[:, 1]
    features[:, 4] = X[:, 0] ** 2
    features[:, 5] = X[:, 0] ** 3
    features[:, 6] = X[:, 0] ** 4

    return features
transformer_opt = FunctionTransformer(opt_features_extract)

def Vop_features_extract(X):
    # Vop = a * exp(T%+10) + b * cell_gap + c
    features = np.empty(shape=(len(X), 3), dtype=float)
    features[:, 0] = 1
#     features[:, 1] = X[:, 0]
    features[:, 1] = X[:, 1]
#     features[:, 3] = X[:, 0] * X[:, 1]
#     features[:, 4] = X[:, 0] ** 2
    features[:, 2] = np.exp(X[:, 0]+10)

    return features
transformer_Vop = FunctionTransformer(Vop_features_extract)

# make a cut off at max T% of opt data
opt_cutoff = pd.DataFrame(columns=opt.columns)
for ID in opt.ID.unique():
    for Point in opt.Point.unique():
        tmp_df = opt[(opt.ID == ID) & (opt.Point == Point)]
        tmp_df = tmp_df.iloc[:tmp_df["LCM_Y%"].argmax(),:]
        opt_cutoff = pd.concat([opt_cutoff, tmp_df])

# check is there axo data
if len(axo) != 0:
    opt_cell_gap = pd.merge(opt_cutoff, axo[["ID", "Point", "cell gap"]], how="left", on=["ID", "Point"])
else:
    opt_cell_gap = pd.merge(opt_cutoff, rdl[["ID", "cell gap"]], how="left", on="ID")

model["opt"] = {}
df = opt_cell_gap.copy()
# some mapping and rename
df["T%"] = opt_cell_gap.groupby(by=["ID", "Point"])["LCM_Y%"].apply(lambda x: 100*x / float(x.max()))
df["Vop"] = df["Voltage"]/2.0
df["LC%"] = df["LCM_Y%"]
df["Wx"] = df["W_x"]
df["Wy"] = df["W_y"]
# the varient is large when Vop is low, so I cut-off at Vop = 2
df = df[df["Vop"] > 3]
training_set, test_set = train_test_split(
    df,
    test_size = 0.2,
#     random_state = 42
)

for LC in cond["LC"].unique():
    model["opt"][LC] = {}
    X_train = training_set[training_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    X_test = test_set[test_set["LC"]==LC][["Vop", "cell gap"]].to_numpy()
    valid_data = [[ref_Vop, ref_cell_gap]]
    for item in ["T%", "LC%"]:
        y_train = training_set[training_set["LC"]==LC][item].to_numpy()
        y_test = test_set[test_set["LC"]==LC][item].to_numpy()
        model["opt"][LC][f'{item}_LR'] = Pipeline([
            ('Scalar', StandardScaler()),
#             ('poly', PolynomialFeatures(degree=3)),
            ('Custom_Transformer', transformer_opt),
            ('linear', linear_model.TheilSenRegressor(fit_intercept=False)),
#             ('linear', linear_model.LinearRegression(fit_intercept=False)),
#             ("GR", GaussianProcessRegressor(kernel=DotProduct()+WhiteKernel()))
        ]).fit(
            X_train, y_train,
        )
        # print(f'R2_test {model["opt"][LC][f"{item}_LR"].score(X_test, y_test):.2f}')
        # ans = float(model["opt"][LC][f"{item}_LR"].predict(valid_data))
        # print(f"{LC}: {item}: {ans:.4f}")
        # print()
    for item in ["Wx", "Wy", "WX", "WY", "WZ"]:
        y_train = training_set[training_set["LC"]==LC][item].to_numpy()
        y_test = test_set[test_set["LC"]==LC][item].to_numpy()
        model["opt"][LC][f'{item}_LR'] = Pipeline([
            ('Scalar', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2)),
#             ('Custom_Transformer', transformer),
            ('linear', linear_model.TheilSenRegressor(fit_intercept=False))
        ]).fit(
            X_train, y_train,
        )
        # print(f'R2_test {model["opt"][LC][f"{item}_LR"].score(X_test, y_test):.2f}')
        # ans = float(model["opt"][LC][f"{item}_LR"].predict(valid_data))
        # print(f"{LC}: {item}: {ans:.4f}")
        # print()

# find V%
# f(T%, cell_gap) -> V%
valid_data = [[90.0, ref_cell_gap]]
training_set_Vop = training_set[training_set["T%"]>89]
test_set_Vop = test_set[test_set["T%"]>89]

for LC in cond["LC"].unique():
    X_train = training_set_Vop[training_set_Vop["LC"]==LC][["T%", "cell gap"]].to_numpy()
    X_test = test_set_Vop[test_set_Vop["LC"]==LC][["T%", "cell gap"]].to_numpy()
    y_train = training_set_Vop[training_set_Vop["LC"]==LC]["Vop"].to_numpy()
    y_test = test_set_Vop[test_set_Vop["LC"]==LC]["Vop"].to_numpy()
    model["opt"][LC][f'Vop_LR'] = Pipeline([
        ('Scalar', StandardScaler()),
#         ('poly', PolynomialFeatures(degree=6)),
        ('Custom_Transformer', transformer_Vop),
        ('linear', linear_model.TheilSenRegressor(fit_intercept=False)),
#         ('linear', linear_model.Linsor(kernel=DotProduct()+WhiteKernel()))
    ]).fit(
        X_train, y_train,
    )
    # print(f'R2_test {model["opt"][LC][f"Vop_LR"].score(X_test, y_test):.2f}')
    # ans = float(model["opt"][LC][f"Vop_LR"].predict(valid_data))
    # print(f"{LC}: Vop: {ans:.4f}")
    # print()

# Generate table
summary_table = pd.DataFrame(
    columns=["LC", "V90", "V95", "V99", "Vmax", "Vop(V)", "Δnd(nm)", "Gap(um)", "LC%", "Wx", "Wx_gain", "Wy", "Wy_gain",
             "u'", "v'", "Δ(u', v')", "a*", "b*", "L*", "Δa*", "Δb*", "ΔL*", "ΔEab*", "CR", "ΔCR(%)", "T%", "Scatter", "D", "W", 
             "Tr(ms)", "Tf(ms)", "RT(ms)", "G2G(ms)"]
)
# cell gap range
# +- 0.5 um, precise to 0.1 um
center_cell_gap = np.round(ref_cell_gap, decimals=1)
cell_gap_range = np.linspace(center_cell_gap-0.5, center_cell_gap+0.5, 11)

# for Eab
def F(X, opt):
    BLU = {
        "Xn": 95.04,
        "Yn": 100.00,
        "Zn": 108.86
    }
    result = 7.787 * X/BLU[opt] + 16/116
    result = result if X/BLU[opt] < 0.008856 else (X/BLU[opt]) ** (1/3)
    return result

# CR
X = [[ref_Vop, ref["cell gap(um)"][0]]]
ref_LC = ref["LC"][0]
ref_CR = ref["CR"][0]
ref_CR_index = ref_CR / (model["opt"][ref_LC]["T%_LR"].predict(X)[0]/float(prop[prop["LC"] == ref_LC]["Scatter index"])/ref["cell gap(um)"][0])
        
for LC in cond["LC"].unique():
    ne = float(prop[prop["LC"] == LC]["n_e"])
    no = float(prop[prop["LC"] == LC]["n_o"])
    scatter_index = float(prop[prop["LC"] == LC]["Scatter index"])
    for cell_gap in cell_gap_range:
        summary_table = summary_table.append({"LC": LC, "Gap(um)": cell_gap}, ignore_index=True)
        # rt
        X = [[ref_Vop, cell_gap]]
        X_minus = [[ref_Vop, cell_gap - 0.1]]
        Tr = model["rt"][LC]["Tr_LR"].predict(X)
        Tf = model["rt"][LC]["Tf_LR"].predict(X)
        RT = model["rt"][LC]["RT_LR"].predict(X)
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "RT(ms)"] = RT[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Tr(ms)"] = Tr[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Tf(ms)"] = Tf[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "G2G(ms)"] = ref["G2G(ms)"][0] / ref["RT(ms)"][0] * RT[0]

        # opt
        Wx = model["opt"][LC]["Wx_LR"].predict(X)
        Wx_gain = Wx - ref["Wx"][0]
        Wy = model["opt"][LC]["Wy_LR"].predict(X)
        Wy_gain = Wy - ref["Wy"][0]
        T = model["opt"][LC]["T%_LR"].predict(X)
        LCp = model["opt"][LC]["LC%_LR"].predict(X)

        Δnd = (ne - no) * cell_gap * 1000
 
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Wx"] = Wx[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Wx_gain"] = Wx_gain[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Wy"] = Wy[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Wy_gain"] = Wy_gain[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "T%"] = T[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "LC%"] = LCp[0] * 100
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Δnd(nm)"] = Δnd
        
        WX = model["opt"][LC]["WX_LR"].predict(X)
        WY = model["opt"][LC]["WY_LR"].predict(X)
        WZ = model["opt"][LC]["WZ_LR"].predict(X)

        Wx_minus = model["opt"][LC]["Wx_LR"].predict(X_minus)
        Wy_minus = model["opt"][LC]["Wy_LR"].predict(X_minus)
        WX_minus = model["opt"][LC]["WX_LR"].predict(X_minus)
        WY_minus = model["opt"][LC]["WY_LR"].predict(X_minus)
        WZ_minus = model["opt"][LC]["WZ_LR"].predict(X_minus)

        # another way to reproduce WX, WZ from xyY
#         WX = [Wx[0] * WY[0] / Wy[0]]
#         WZ = [(1 - Wx[0] - Wy[0]) * WY[0] / Wy[0]]
        
        F_X = F(WX[0], "Xn")
        F_Y = F(WY[0], "Yn")
        F_Z = F(WZ[0], "Zn")
        a_star = 500 * (F_X - F_Y)
        b_star = 200 * (F_Y - F_Z)
        L_star = 116 * F_Y - 16
        u_prime = 4 * Wx[0] / (-2 * Wx[0] + 12 * Wy[0] + 3)
        v_prime = 9 * Wy[0] / (-2 * Wx[0] + 12 * Wy[0] + 3)
        
        F_X_minus = F(WX_minus[0], "Xn")
        F_Y_minus = F(WY_minus[0], "Yn")
        F_Z_minus = F(WZ_minus[0], "Zn")
        a_star_minus = 500 * (F_X_minus - F_Y_minus)
        b_star_minus = 200 * (F_Y_minus - F_Z_minus)
        L_star_minus = 116 * F_Y_minus - 16
        Δa_star = a_star - a_star_minus
        Δb_star = b_star - b_star_minus
        ΔL_star = L_star - L_star_minus
        ΔE_ab_star = (Δa_star**2 + Δb_star**2 + ΔL_star**2)**(1/2)
        u_prime_minus = 4 * Wx_minus[0] / (-2 * Wx_minus[0] + 12 * Wy_minus[0] + 3)
        v_prime_minus = 9 * Wy_minus[0] / (-2 * Wx_minus[0] + 12 * Wy_minus[0] + 3)
        Δuv = ((u_prime - u_prime_minus)**2 + (v_prime - v_prime_minus)**2)**(1/2)
        
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "a*"] = a_star
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "b*"] = b_star
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "L*"] = L_star
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Δa*"] = Δa_star
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Δb*"] = Δb_star
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "ΔL*"] = ΔL_star
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "ΔEab*"] = ΔE_ab_star

        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "u'"] = u_prime
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "v'"] = v_prime
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Δ(u', v')"] = Δuv

        # V%
        V90 = model["opt"][LC]["Vop_LR"].predict([[90, cell_gap]])
        V95 = model["opt"][LC]["Vop_LR"].predict([[95, cell_gap]])
        V99 = model["opt"][LC]["Vop_LR"].predict([[99, cell_gap]])
        Vmax = model["opt"][LC]["Vop_LR"].predict([[100, cell_gap]])
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "V90"] = V90[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "V95"] = V95[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "V99"] = V99[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Vmax"] = Vmax[0]
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Vop(V)"] = ref_Vop
        
        # CR
        Scatter = scatter_index * cell_gap
        D = Scatter
        W = T[0]
        CR = W/D * ref_CR_index
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "D"] = D
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "W"] = W
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "Scatter"] = Scatter
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "CR"] = CR
        summary_table.loc[((summary_table["LC"] == LC) & (summary_table["Gap(um)"] == cell_gap)), "ΔCR(%)"] = (CR-ref_CR)/ref_CR * 100
        

summary_table.to_sql("summary", con=engine, if_exists="append", index=False)
print("calculate complete")