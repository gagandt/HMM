import numpy as np
import pandas as pd
import sys
import os

from dtw_super import DTWSuper


#training data sets
x1_train = []
with open('data/8/train/x1_train.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x1_train.append(seq)

x2_train = []
with open('data/8/train/x2_train.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x2_train.append(seq)

x3_train = []
with open('data/8/train/x3_train.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x3_train.append(seq)

x1_train = np.array(x1_train)
x2_train = np.array(x2_train)
x3_train = np.array(x3_train)

y1_train = np.full(len(x1_train), 1)
y2_train = np.full(len(x2_train), 2)
y3_train = np.full(len(x3_train), 3)


#testing data sets
x = []
y = []

with open('data/8/test/x1_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x.append(seq)
       y.append(1)


with open('data/8/test/x2_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x.append(seq)
       y.append(2)


with open('data/8/test/x3_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x.append(seq)
       y.append(3)

test_run = DTWSuper(x, y, x1_train, x2_train, x3_train, y1_train, y2_train, y3_train, 4)
print(test_run.knn_classifier())


        
        
            