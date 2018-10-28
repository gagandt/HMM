import numpy as np
import sys

""" assuming input is coming in standard np array"""

class Dtw:
    x_test = np.array([])
    x1_train = np.array([])
    x2_train = np.array([])
    x3_train = np.array([])
    y1_train = np.array([])
    y2_train = np.array([])
    y3_train = np.array([])
    k = 0
    
    def __init__(self, x1, x2, x3, y1, y2, y3, knn_para):
        #x_test = x
        self.x1_train = x1
        self.x2_train = x2
        self.x3_train = x3
        self.y1_train = y1
        self.y2_train = y2
        self.y3_train = y3
        self.k = knn_para
    
    def dtw(self, x_train, label, sequence, k):
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
                    cost = abs(sequence[i-1] - test[j-1])
                    mat[i][j] = cost + min(mat[i-1][j], mat[i-1][j-1], mat[i][j-1])
            
            knn_array = np.append(knn_array, mat[n-1][m-1])
        
        knn_array.sort(axis = 0)
        #print(knn_array)
        
        return knn_array[:k]
    
    def knn(self, sequence):
        knn_array = np.array([])
        knn_class = np.array([])
        knn_array = self.dtw(self.x1_train, 1, sequence, self.k)
        t1 = np.empty(self.k); t1.fill(1)
        knn_class = np.append(knn_class, t1)
        
        knn_array = np.append(knn_array, self.dtw(self.x2_train, 2, sequence, self.k))
        t1 = np.empty(self.k); t1.fill(2)
        knn_class = np.append(knn_class, t1)
        #knn_array.sort(axis = 0)
        #knn_array = knn_array[:self.k]
        knn_array = np.append(knn_array, self.dtw(self.x3_train, 2, sequence, self.k))
        t1 = np.empty(self.k); t1.fill(3)
        knn_class = np.append(knn_class, t1)
        
        knn_array, knn_class = zip(*sorted(zip(knn_array, knn_class)))
        #knn_array.sort(axis = 0)
        knn_array = knn_array[:self.k]
        knn_class = knn_class[:self.k]
        
        count = [0,0,0]
        for i in range(0,self.k):
            count[int(knn_class[i])-1] += 1
        if (count[0] == max(count[0], count[1], count[2])):
            return 1
        elif (count[1] == max(count[0], count[1], count[2])):
            return 2
        else:
            return 3

x1 = [[1,2,3,1,2,3], [1,2,3,1,3], [1,2,1,2,3]]
y1 = [1,1,1]
x2 = [[3,2,1,3,2,1], [3,2,1,3,2], [3,1,3,2,1]]
y2 = [2,2,2]
x3 = [[1,3,2,1,3,2], [1,3,2,1,2], [1,2,1,3,2]]
y3 = [3,3,3]

test_run = Dtw(x1, x2, x3, y1, y2, y3,3)
print(test_run.knn([1,3,2,1,3,2]))
        
        
            