import os, sys

# all parameter from args in .ts file
params_str = {
    "argv1": sys.argv[1]
    #...
}

# Cause it execute by the top app.js, the work directory is at the root directory
# You can check by the following line.
print("current work directory:", str(os.getcwd())) # root should in path_to_project/tdtoolkit_web

# do your thing here
# You can copy the code directory from test.ipynb at root directory

print(params_str['argv1'])

# all print would be the output of pythonShell in .ts file
print('template success')