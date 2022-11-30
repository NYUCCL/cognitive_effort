
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

from LaTeXDatax import datax
# Information on this package can be found at https://github.com/Datax-package/LaTeXDatax.py
# NOTE: i edited this package 8/18/2022 to take dictionary input rather than **vars


# import regression results from csvs
pathtofiles = "regression_results/"
resultsfiles = [f for f in listdir(pathtofiles) if isfile(join(pathtofiles, f)) and not f.startswith(".")]
# print(resultsfiles)

allresults = {} # big dictionary of informatively labeled variables

for file in resultsfiles:
    label = file[0:-4]
    # print(label)

    # for each file, read, turn into dictionary and then unpack all the values to long string variablenames 
    superdict = pd.read_csv(join(pathtofiles, file),index_col=[0]).to_dict()
    # print(superdict)

    for Skey, SVal in superdict.items():
        for key, val in SVal.items():
            varname = label + "_" + key + "_" + Skey # e.g., results0_Intercept_mean
            
            # delete % because this symbol messes up the tex file!
            varname = varname.replace("%","")

            # print(varname)
            allresults[varname] = val
            print(varname + " = " + str(val))

# print(allresults)

# save allresults in data.tex
# allresults={"a":1,"b":2}
datax(allresults, filename = "data.tex")
