import numpy as np
import pandas as pd
import sys
import os

from dtw_super import DTWSuper
from data_initialiser import DATA
f = 4;
for i in range(1, 4):     
    f *= 2
    d1 = DATA(8,f)
    d2 = DATA(16,f)
    d3 = DATA(32,f)
    print("for k = 8, KNN k  = " + str(f))
    d1.fit()
    print("for k = 16, KNN k  = " + str(f))
    d1.fit()
    print("for k = 32, KNN k  = " + str(f))
    d3.fit()

