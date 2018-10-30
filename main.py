import numpy as np
import pandas as pd
import sys
import os

from dtw_super import DTWSuper
from data_initialiser import DATA

d1 = DATA(8,4)
d2 = DATA(16,4)
d3 = DATA(32,4)

d1.fit()
d1.fit()
d3.fit()