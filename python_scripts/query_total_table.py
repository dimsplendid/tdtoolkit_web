from os import write
import sys
import time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sqlalchemy as sql
from openpyxl import load_workbook
# import os

params_str = {
    "V%": sys.argv[1],
    "LC": sys.argv[2],
    "cell_gap_lower": sys.argv[3],
    "cell_gap_upper": sys.argv[4],
}
# test input
# params_str = {
#     "LC": "LCT-19-580,MOX-1",
#     "cell_gap_lower": "2.1,2.1",
#     "cell_gap_upper": "3.0,3.0",
# }

engine = sql.create_engine('sqlite:///./database/test.db', echo=False)

params = {}
for k, v in params_str.items():
    params[k] = v.split(",")

# print(os.getcwd())

result_df = pd.DataFrame()
# output = './public/tmp/'
output = './tmp/'
rnd_file_code = f"-{np.random.randint(0, 10000):04d}"
file_name = output + 'query-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + rnd_file_code

for i in range(len(params["LC"])):
#     print(params['LC'][i])
    tmp_df = pd.read_sql(f"SELECT * FROM summary WHERE LC == \"{params['LC'][i]}\" AND \"Gap(um)\" > {params['cell_gap_lower'][i]} AND \"Gap(um)\" < {params['cell_gap_upper'][i]} AND \"V%\" == \'{params['V%'][0]}\'", engine)
    if (params['V%'][0] == 'Vref'):
        tmp_ref = pd.read_sql(f"SELECT * FROM ref", engine)
        # rename ref column
        tmp_ref.columns = [c + '_ref' for c in tmp_ref.columns]
        tmp_df = tmp_df.merge(tmp_ref, how="left", left_on="Batch", right_on="batch_ref")
    result_df = pd.concat([result_df, tmp_df], ignore_index=True)

result_df.to_excel(file_name + '.xlsx', sheet_name="opt")

VT_df = pd.DataFrame()
for i in range(len(params["LC"])):
#     print(params['LC'][i])
    tmp_df = pd.read_sql(f"SELECT * FROM VT_CURVE WHERE LC == \"{params['LC'][i]}\" AND \"Gap(um)\" > {params['cell_gap_lower'][i]} AND \"Gap(um)\" < {params['cell_gap_upper'][i]}", engine)
    VT_df = pd.concat([VT_df, tmp_df], ignore_index=True)

book = load_workbook(file_name + '.xlsx')
writer = pd.ExcelWriter(file_name + '.xlsx', engine='openpyxl')
writer.book = book
VT_df.to_excel(writer, sheet_name="VT_curve")
writer.save()
writer.close()
# print(params)
print(file_name + ".xlsx")
# print('python finished')