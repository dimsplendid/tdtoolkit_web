import sys
import time
import numpy as np
import os
import pandas as pd

# from tdtoolbox.axo import axo_load

# read file(s)
axo_path = sys.argv[1]
print(axo_path)
# axo_path = r'D:\Projects\tdtoolkit_web\examples\axo-exporter-example'
# set cut off
rms_cutoff = float(sys.argv[2])
# rms_cutoff = 0.3


def axo_load(path):
    df = pd.DataFrame()
    # may wrong, need caution
    for cwd, dir_name, file_names in os.walk(path):
        for f in file_names:
            file = os.path.join(cwd, f)
            
            tmp_df = pd.read_csv(file, engine="python", skiprows=27, skipfooter=92)
            tmp_df.insert(loc = 0, column = "file name", value=f)
            # some data has different title(?), so we rename it to make concat well 
            tmp_df.columns = ["file name", "Chip No.", "x", "y", "cell gap", "top rubbing direct", "twist", "top pre-tilt", "bottom pre-tilt", "rms", "iteration"]
            df = pd.concat([df, tmp_df], ignore_index=True)
    
    return df
axo = axo_load(axo_path)
axo = axo[axo["rms"] < rms_cutoff]

# save file
output = './tmp/'
rnd_file_code = f"-{np.random.randint(0, 10000):04d}"
file_name = output + 'axoexport-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + rnd_file_code

axo.to_excel(file_name + '.xlsx')

# output
print(file_name + '.xlsx')