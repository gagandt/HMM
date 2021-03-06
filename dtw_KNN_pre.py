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
        knn_array = np.array([100, label])
        #print(knn_array)

        for test in x_train:
            m = len(test) + 1
            #   n X m
            mat = [[0 for x in range(m)] for y in range(n)] 
            
            for i in range(0, n):
                mat[i][0] = 100
                
            for i in range(0, m):
                mat[0][i] = 100
            
            mat[0][0] = 0
            
            for i in range(1, n):
                for j in range(1, m):
                    cost = abs(sequence[i-1] - test[j-1])
                    mat[i][j] = cost + min(mat[i-1][j], mat[i-1][j-1], mat[i][j-1])
            
            
            elem = np.array([mat[n-1][m-1], label])
            #print(elem)
            knn_array = np.vstack((knn_array, elem))
            #knn_array = np.insert(knn_array, 1, [mat[n-1][m-1], label], axis = 1)
            #print (mat)
        
        knn_array.sort(axis = 0)
        
        #print(knn_array)
        #print("hi")
        
        return knn_array[:k]
    
    def knn(self, sequence):
        knn_array = np.array([])
        knn_array = self.dtw(self.x1_train, 1, sequence, self.k)
        knn_array = np.vstack((knn_array, self.dtw(self.x2_train, 2, sequence, self.k)))
        knn_array.sort(axis = 0)
        #knn_array = knn_array[:self.k]
        knn_array = np.vstack((knn_array, self.dtw(self.x3_train, 3, sequence, self.k)))
        knn_array.sort(axis = 0)
        #knn_array = knn_array[:self.k]

        
        count = [0,0,0]
        #for seq in knn_array:
         #   count[seq[1]-1] += 1
        print(knn_array[0][1])
        
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
print(test_run.knn([3,2,1,3,2,1]))

#Here we have the stored the KNN as both the dtw distance and the class assigned in a 2d vector.
#Can build a sort function to it and sort accordingly!
        
        
            