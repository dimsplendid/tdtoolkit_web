import sys
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

params_str = {
    "LC": sys.argv[1],
    "cell_gap": sys.argv[2],
    "V_max": sys.argv[3],
    "V_min": sys.argv[4]
}
# test input
# params = {
#     "LC": "LCT-19-580, MOX-1",
#     "cell_gap": "2.5, 2.1",
#     "V_max": "10.0, 10.0",
#     "V_min": "2.0, 2.0"
# }
params = {}
for k, v in params_str.items():
    params[k] = v.split(",")



result_df = pd.DataFrame()
output = './public/tmp/'
rnd_file_code = f"-{np.random.randint(0, 10000):04d}"
file_name = output + 'query-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + rnd_file_code
for i in range(len(params["LC"])):
    LC = params["LC"][i]
    path = './notebooks/'+ f'{LC}-VT.h5'

    model = load_model(path)
    # model.summary()
    cell_gap = float(params["cell_gap"][i])
    V_min = float(params["V_min"][i])
    V_max = float(params["V_max"][i])

    query_V = np.linspace(V_min, V_max, 100)
    X_query = np.array([[v, cell_gap] for v in query_V])
    # print("X_query.shape: ", X_query.shape)
    y_query = model.predict(X_query)

    plt.scatter(X_query[:,0], y_query, label=LC)
    tmp_df = pd.DataFrame(X_query, columns=["Vop", "cell gap"])
    tmp_df.insert(0, "LC", LC)
    tmp_df["T%"] = y_query
    tmp_df["model"] = path
    result_df = pd.concat([result_df, tmp_df], ignore_index=True)
plt.legend()
plt.xlabel('volt')
plt.ylabel('T%')
plt.title('V-T curve')
plt.savefig(file_name + '.png')
result_df.to_excel(file_name + '.xlsx')

print(params)
print('python finished')