import sys
import time
import numpy as np
import os
import pandas as pd

from tdtoolbox.axo import axo_load

# read file(s)
axo_path = sys.argv[1]
print(axo_path)
# axo_path = r'D:\Projects\tdtoolkit_web\examples\axo-exporter-example'
# set cut off
# rms_cutoff = float(sys.argv[2])
rms_cutoff = 0.5

axo = axo_load(axo_path)
axo = axo[axo["rms"] < rms_cutoff]

# save file
output = './tmp/'
rnd_file_code = f"-{np.random.randint(0, 10000):04d}"
file_name = output + 'axoexport-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + rnd_file_code

axo.to_excel(file_name + '.xlsx')

# output
print(file_name + '.xlsx')