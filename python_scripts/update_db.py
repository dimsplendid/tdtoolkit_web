import os
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import String
import shutil

# print("file path:", os.getcwd()) # root should in ./tdtoolkit_web
raw_root = os.path.join('.','src','app', 'raw')

path = {
    "axo": os.path.join(raw_root, "AXO"),
    "rdl": os.path.join(raw_root, "RDL"),
    "opt": os.path.join(raw_root, "OPT"),
    "rt": os.path.join(raw_root, "RT"),
    "cond": os.path.join(raw_root, "CONDITIONS"),
    "prop": os.path.join(raw_root, "PROPERTY"),
    "ref": os.path.join(raw_root, "REF"),
    "output": "output",
    "db": "database"
}

# need to separate later
def axo_load(path, cond=pd.DataFrame()):
    """
    The path and the file name need to be below:
    Could be multiple file
    ─┬─ folder[path]
     ├─ 1-1 + 1-2 + 1-3.csv
     ├─ 1-4.csv
     ...
    """
    df = pd.DataFrame()
    # may wrong, need caution
    loc = [5, 3, 1, 6, 4, 2]
    for cwd, dir_name, file_names in os.walk(path):
        for f in file_names:
            file = os.path.join(cwd, f)
            # get short-id from file name
            # Cause there may be multiple panel in one file, we need to
            # deal with it.
            short_id = f.split(".")[0]
            short_id = [s.strip() for s in short_id.split("+")]
            # multiple each point 6 times cause there are 6 point in one panel
            # maybe I should split this function to another place?
            short_id_6 = [id for id in short_id for _ in range(6)]
            location_6 = loc * len(short_id)
            # Todo: more error format handling
            
            tmp_df = pd.read_csv(file, engine="python", skiprows=27, skipfooter=92)
            tmp_df.insert(loc = 1, column = "short-id", value=short_id_6)
            tmp_df.insert(loc = 2, column = "point", value = location_6)
            # some data has different title(?), so we rename it to make concat well 
            tmp_df.columns = ["Chip No.", "Short-id",  "Point", "x", "y", "cell gap", "top rubbing direct", "twist", "top pre-tilt", "bottom pre-tilt", "rms", "iteration"]
            df = pd.concat([df, tmp_df], ignore_index=True)
    # replace short-id if you have condition table
    if cond.empty != True:
        df = df.rename(columns={"Short-id": "ID", "Chip No.": "LC"})
        df["ID"] = df["ID"].map(dict(cond[["Short-id", "ID"]].values))
        df["LC"] = df["ID"].map(dict(cond[["ID", "LC"]].values))
        df["Project"] = df["ID"].map(dict(cond[["ID", "Project"]].values))
        df["Batch"] = df["ID"].map(dict(cond[["ID", "Batch"]].values))
        # neglect the data that doesn't record
        df = df[~df["ID"].isna()]
        df.columns = ["LC", "ID", "Point", "x", "y", "cell gap", "top rubbing direct", "twist", "top pre-tilt", "bottom pre-tilt", "rms", "iteration", "Project", "Batch"]

    return df

def rdl_load(path, cond=pd.DataFrame()):
    """
    The path and the file name need to be below:
    Should be single file.
    ─┬─ folder[path]
     └─ [cell gap].xlsx
    """
    file = next(os.walk(path))[2][0]
    df = pd.read_excel(os.path.join(path, file))
    if cond.empty != True:
        df = df.rename(columns={"Short-id": "ID"})
        df["ID"] = df["ID"].map(dict(cond[["Short-id", "ID"]].values))
        df["LC"] = df["ID"].map(dict(cond[["ID", "LC"]].values))
        df["Project"] = df["ID"].map(dict(cond[["ID", "Project"]].values))
        df["Batch"] = df["ID"].map(dict(cond[["ID", "Batch"]].values))
        # neglect the data that doesn't record
        df = df[~df["ID"].isna()]
        df.columns = ['ID', 'cell gap', 'LC', "Project", "Batch"]
    return df

def opt_load(path, cond=pd.DataFrame()):
    """
    The path and the file name need to be below:
    Could be multiple file
    ─┬─ folder[path]
     ├─ xxxx.csv
     ├─ xxxx.csv
     ...
    """
    df = pd.DataFrame()
    for cwd, dir_name, file_names in os.walk(path):
        for f in file_names:
            if f[0] == ".":
                continue
            if f[0] == "~":
                continue
            file = os.path.join(cwd, f)
            tmp_df = pd.read_csv(file, encoding="ansi").iloc[:,:64]
            # some data preprocessing
            df = pd.concat([df, tmp_df], ignore_index=True)
            
    df.columns = ['Data', 'M_Time', 'ID', 'Point', 'Station', 'Operator', 'Voltage',
        'I.Time', 'AR_T%(⊥)', 'AR_T%(//)', 'LCM_X%', 'LCM_Y%', 'LCM_Z%', 'RX',
        'RY', 'RZ', 'GX', 'GY', 'GZ', 'BX', 'BY', 'BZ', 'WX', 'WY', 'WZ', 'CG%',
        'R_x', 'R_y', 'G_x', 'G_y', 'B_x', 'B_y', 'W_x', 'W_y', 'RX_max',
        'GY_max', 'BZ_max', 'V_RX_max', 'V_GY_max', 'V_BZ_max', "WX'", "WY'",
        "WZ'", "W_x'"," W_x'.1", 'LCM_X%max', 'LCM_Y%max', 'LCM_Z%max',
        'φ_(Ymax)', 'φ_(Ymax).1', 'φ_(Zmax)', 'φ_tol_X', 'φ_tol_Y', 'φ_tol_Z',
        'T0/Tmax_X', 'T0/Tmax_Y', 'T0/Tmax_Z', 'Vcri_X', 'Vcri_Y', 'Vcri_Z',
        'dφ_X', 'dφ_Y', 'dφ_Z', "LC"]
    # voltage == 1 is the wrong rows, need drop
    df = df[df["Voltage"] != 1]
    if cond.empty != True:
        df["LC"] = df["ID"].map(dict(cond[["ID", "LC"]].values))
        df["Project"] = df["ID"].map(dict(cond[["ID", "Project"]].values))
        df["Batch"] = df["ID"].map(dict(cond[["ID", "Batch"]].values))
    return df

def rt_load(path, cond=pd.DataFrame()):
    """
    The path and the file name need to be below:
    Could be multiple file
    ─┬─ folder[path]
     ├─ xxxx.txt
     ├─ xxxx.txt
     ...
    """
    df = pd.DataFrame()
    for cwd, dir_name, file_names in os.walk(path):
        for f in file_names:
            if f[0] == ".":
                continue
            if f[0] == "~":
                continue
            file = os.path.join(cwd, f)
            tmp_df = pd.read_table(file, encoding="ansi")
            # some system encoding would go wrong, so I rename here
            # there are some implicit problem, the data should just like this
            tmp_df.columns = ['Date', 'Time', 'ID', 'Point', 'Station', 'Operator', 'cell pos.', 'Target Vpk',
               'Initial Vpk', 'OD_Rise', 'OD_fall', 'Normalized_V', 'Specific_target',
               'Photo Sensor', 'TempSensor', 'Temp', 'Model', 'Rise-mean (10-90)',
               'Rise-stdev (10-90)', 'Fall-mean (10-90)', 'Fall-stdev (10-90)',
               'Rise-mean (5-95)', 'Rise-stdev (5-95)', 'Fall-mean (5-95)',
               'Fall-stdev (5-95)', 'Vcom', 'Flicker', 'Base lv-mean', 'Top lv-mean',
               'WXT (%)', 'BXT (%)', 'WXT_*', 'BXT_*', 'Overshooting or not',
               'Overshooting %', 'TailTime', 'overshooting_peak', 'overshooting_top',
               '(RisePeak-top)/top', '(FallPeak-base)/base', 'delta_peak', 'delta_v',
               'delta_m', 'c_a', 'peak', 'top', 'HLH_(Peak-Top)', 'HLH_area']
            # neglect incorrect rows
            tmp_df = tmp_df[tmp_df["Point"].isin([1, 2, 3, 4, 5, 6, '1', '2', '3', '4', '5', '6'])]
            # correct the data types
            for col in tmp_df.columns:
                try:
                    tmp_df[col] = tmp_df[col].astype('float')
                except:
                    continue
            df = pd.concat([df, tmp_df], ignore_index=True)
    
    if cond.empty != True:
        df["LC"] = df["ID"].map(dict(cond[["ID", "LC"]].values))
        df["Project"] = df["ID"].map(dict(cond[["ID", "Project"]].values))
        df["Batch"] = df["ID"].map(dict(cond[["ID", "Batch"]].values))
    df = df[df["ID"]!="NAN"]
    
    return df

def cond_load(path):
    """
    The path and the file name need to be below:
    Should be single file.
    ─┬─ folder[path]
     └─ [cond].xlsx
    """
    file = next(os.walk(path))[2][0]
    df = pd.read_excel(os.path.join(path, file))
    df = df.iloc[:,0:5]
    df.columns = ["ID", "LC", "Short-id", "Project", "Batch"]
    return df

def prop_load(path):
    """
    The path and the file name need to be below:
    Should be single file.
    ─┬─ folder[path]
     └─ [prop].xlsx
    """
    file = next(os.walk(path))[2][0]
    df = pd.read_excel(os.path.join(path, file))
    df['Scatter index'] = (df['n_e'] ** 2 - df['n_o'] ** 2) ** 2 * 3 / (df['K11(pN)'] + df['K22(pN)'] + df['K33(pN)'])
    df['RT index'] = df['rotation viscosity (γ1)(mPa⋅s)'] / df['K22(pN)']
    return df

def ref_load(path):
    """
    The path and the file name need to be below:
    Should be single file.
    ─┬─ folder[path]
     └─ [ref].xlsx
    """
    file = next(os.walk(path))[2][0]
    df = pd.read_excel(os.path.join(path, file))
    return df

# loading data from raw
cond = cond_load(path["cond"])
axo = axo_load(path["axo"], cond)
rdl = rdl_load(path["rdl"], cond)
opt = opt_load(path["opt"], cond)
rt = rt_load(path["rt"], cond)
prop = prop_load(path["prop"])
ref = ref_load(path["ref"])


# writing to database

engine = sql.create_engine('sqlite:///database/test.db', echo=False)
# engine = sql.create_engine('sqlite://', echo=True)

# Would occur strange ID conflic(although not duplicate), need check
# meta = sql.MetaData()
# # Model 是否要分出去?
# sql.Table(
#     "cond", meta,
#     sql.Column("LC", sql.String),
#     sql.Column("Short-id", sql.String),
#     sql.Column("ID", sql.String, unique=True),
#     sql.Column("Project", sql.String),
#     sql.Column("Batch", sql.String)
# )

# meta.create_all(engine)

## check constrain
try:
    cond.to_sql("cond", con=engine, if_exists="append", index=False)
    axo.to_sql("axo", con=engine, if_exists="append", index=False)
    rdl.to_sql("rdl", con=engine, if_exists="append", index=False)
    opt.to_sql("opt", con=engine, if_exists="append", index=False)
    rt.to_sql("rt", con=engine, if_exists="append", index=False)
    prop.to_sql("prop", con=engine, if_exists="append", index=False)
    ref.to_sql("ref", con=engine, if_exists="append", index=False)
    shutil.rmtree(raw_root)
    print("Database update!")
    print(cond.Batch[0])
except:
    print("Something wrong, maybe the ID condition is duplicate. Database keep")