import numpy as np
import pandas as pd
import sys
import os

from dtw_super import DTWSuper

""" assuming input is coming in standard np array"""
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



x = []
y = []

with open('data/8/test/x3_test.txt') as f:
   for line in f:
       seq = [int(x) for x in line.split()]
       x.append(seq)
       y.append(3)

test_run = DTWSuper(x, y, x1, x2, x3, y1, y2, y3, 5)
print(test_run.knn_classifier())


"""
fc = open("data/x1_train.txt")
mylist = str.split(readlines(fc), " ")
x1 = list(map(int, mylist))

fc <- file("data/x2_train.txt")
mylist <- strsplit(readLines(fc), " ")
x2 = list(map(int, mylist))

fc <- file("data/x3_train.txt")
mylist <- strsplit(readLines(fc), " ")
x3 = list(map(int, mylist))



close(fc)


fi = open("data/x2_train.txt")
fo = open("output.dat", "w")
x1 = []

for line in fi:
    line = line.strip()
    x1.append(line.split())


print(x1[0])

y1 = np.empty(len(x1)); y1.fill(1)
y2 = np.empty(len(x2)); y2.fill(2)
y3 = np.empty(len(x3)); y3.fill(3)


x1 = [[1,2,3,1,2,3], [1,2,3,1,3], [1,2,1,2,3]]
y1 = [1,1,1]
x2 = [[3,2,1,3,2,1], [3,2,1,3,2], [3,1,3,2,1]]
y2 = [2,2,2]
x3 = [[1,3,2,1,3,2], [1,3,2,1,2], [1,2,1,3,2]]
y3 = [3,3,3]


"""
        
        
            