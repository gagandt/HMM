import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

#   DTW Super Class.
#   Solution to the first Question.
#   Test Data and the Training Data are needed and the K value has to be specified.
#   KNN is used for classification and the distance measure used is the DTW distance.

class DTWSuper:
    #Class Data
    x_test = np.array([])
    y_test = np.array([])
    x1_train = np.array([])
    x2_train = np.array([])
    x3_train = np.array([])
    y1_train = np.array([])
    y2_train = np.array([])
    y3_train = np.array([])
    k = 0
    
    #Initialising 
    #Testing Data : x_test; Testing Classes : y_test
    #Training Data : x1, x2, x3; Training Classes : y1, y2, y3
    #K for KNN : knn_para
    def __init__(self, x_test, y_test, x1, x2, x3, y1, y2, y3, knn_para):
        self.x_test = x_test
        self.y_test = y_test
        self.x1_train = x1
        self.x2_train = x2
        self.x3_train = x3
        self.y1_train = y1
        self.y2_train = y2
        self.y3_train = y3
        self.k = knn_para
    
    #Discrete Time Warping method for audio sequences.
    #Training Class : x_train
    #Test Sequence : sequence
    #k for KNN : k
    def dtw(self, x_train, sequence, k):
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
            '''
            distance, path = fastdtw(test, sequence, dist=euclidean)
            knn_array = np.append(knn_array, distance)
            '''
           
        knn_array.sort(axis = 0)
        #print(knn_array)
        
        return knn_array[:k]
    
    #KNN Classifier for a Sequence
    def knn(self, sequence):
        knn_array = np.array([])
        knn_class = np.array([])

        knn_array = self.dtw(self.x1_train, sequence, self.k)
        t1 = np.empty(self.k); t1.fill(1)
        knn_class = np.append(knn_class, t1)
        
        knn_array = np.append(knn_array, self.dtw(self.x2_train, sequence, self.k))
        t1 = np.empty(self.k); t1.fill(2)
        knn_class = np.append(knn_class, t1)
        
        knn_array = np.append(knn_array, self.dtw(self.x3_train, sequence, self.k))
        t1 = np.empty(self.k); t1.fill(3)
        knn_class = np.append(knn_class, t1)
        
        knn_array, knn_class = zip(*sorted(zip(knn_array, knn_class)))
        knn_array = knn_array[:self.k]
        knn_class = knn_class[:self.k]
        
        #print(knn_class)
        #print("-")
        count = [0,0,0]
        for i in range(0,self.k):
            count[int(knn_class[i])-1] += 1

        if (count[0] == max(count[0], count[1], count[2])):
            return 1
        elif (count[1] == max(count[0], count[1], count[2])):
            return 2
        else:
            return 3

    #KNN Classifier for the Testing Data
    def knn_classifier(self):
        y_pred = np.array([])

        for test in self.x_test:
            out_class = self.knn(test)
            y_pred = np.append(y_pred, out_class)

       # print(y_pred)
        #print(len(y_pred))
        #print(len(self.y_test))
        M = confusion_matrix(self.y_test, y_pred)
        print(M)

        ret = float(self.knn_utility(M))
        return ret

    #Utility function for Precision, Recall and F-Measure.
    def knn_utility(self, M):
        hsum = [0,0,0]
        r = [0,0,0]
        p = [0,0,0]

        for j in range(0,3):
            h = 0
            v = 0;
            for i in range(0,3):
                h += M[j][i];
                v += M[i][j];
            
            hsum[j] = h;
            #r[j] = M[j][j]*1.0/h*1.0
            #p[j] = arr[j][j]*1.0/v*1.0
        
        return (float(M[0][0] + M[1][1] + M[2][2])/float(hsum[1] + hsum[2] + hsum[0]))
