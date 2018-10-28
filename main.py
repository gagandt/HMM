import numpy as np
import sys
import dtw

""" assuming input is coming in standard np array"""



x1 = [[1,2,3,1,2,3], [1,2,3,1,3], [1,2,1,2,3]]
y1 = [1,1,1]
x2 = [[3,2,1,3,2,1], [3,2,1,3,2], [3,1,3,2,1]]
y2 = [2,2,2]
x3 = [[1,3,2,1,3,2], [1,3,2,1,2], [1,2,1,3,2]]
y3 = [3,3,3]

test_run = DTW(x1, x2, x3, y1, y2, y3,3)
print(test_run.knn([1,3,2,1,3,2]))
        
        
            