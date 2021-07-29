import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as sql
import os

params_str = {
    "LC": sys.argv[1],
    "cell_gap_lower": sys.argv[2],
    "cell_gap_upper": sys.argv[3],
}
# test input
# params_str = {
#     "LC": "LCT-19-580,MOX-1",
#     "cell_gap_lower": "2.1,2.1",
#     "cell_gap_upper": "3.0,3.0",
# }

engine = sql.create_engine('sqlite:///./database/demo2.db', echo=False)

params = {}
for k, v in params_str.items():
    params[k] = v.split(",")

print(os.getcwd())

result_df = pd.DataFrame()
# output = './public/tmp/'
output = './tmp/'
rnd_file_code = f"-{np.random.randint(0, 10000):04d}"
file_name = output + 'query-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + rnd_file_code
for i in range(len(params["LC"])):
    print(params['LC'][i])
    tmp_df = pd.read_sql(f"SELECT * FROM summary WHERE LC == \"{params['LC'][i]}\" AND \"Gap(um)\" > {params['cell_gap_lower'][i]} AND \"Gap(um)\" < {params['cell_gap_upper'][i]}", engine)
    result_df = pd.concat([result_df, tmp_df], ignore_index=True)

result_df.to_excel(file_name + '.xlsx')

print(params)
print("file save in ", file_name + ".xlsx")
print('python finished')