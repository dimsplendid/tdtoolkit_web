import os
import pandas as pd

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