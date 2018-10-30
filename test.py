#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:49:54 2018

@author: gdt
"""

import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

x = np.array([1,2,3,4,5])
y = np.array([2,3,5])
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)

x_train = np.array([])
x_train = [[2,3,4,5], [1,2,3,4,5]]
sequence = y

def dtw(x_train, sequence, k):
        n = len(sequence) + 1
        knn_array = np.array([])

        for test in x_train:
            m = len(test) + 1
            #   n X m
            mat = [[0 for x in range(m)] for y in range(n)] 
            
            for i in range(0, n):
                mat[i][0] = sys.maxsize    
            for i in range(0, m):
                mat[0][i] = sys.maxsize
            mat[0][0] = 0
            
            for i in range(1, n):
                for j in range(1, m):
                    cost = abs(sequence[i-1] - int(test[j-1]))
                    mat[i][j] = cost + min(mat[i-1][j], mat[i-1][j-1], mat[i][j-1])
            
            knn_array = np.append(knn_array, mat[n-1][m-1])
        
        knn_array.sort(axis = 0)
        #print(knn_array)
        
        return knn_array[:k]

print(dtw(x,y,7))