import numpy as np



def DTWminDist(mat, x,y,dist):
    
    if x == mat.shape[0] and y == mat.shape[1]:
        return dist
    if y == mat.shape[1]:
        dist = dist + np.abs(mat[x,y] - mat[x+1,y])
        D(mat, x+1, y, dist)

   

class DTW:
    
    def getDistMat(self, testItem, val):
        distMat = np.zeros((len(testItem),len(val)))
        for i in range(0, len(testItem)):
            for j in range(0, len(val)):
                distMat[i,j] = np.abs(testItem[i]-val[j])

        return distMat

    def __init__(self, **kwargs):
        self.templates = kwargs.items()
        self.model = NULL
        self.minDist = 10e18
    
    def fit(self, testItem):
        
        for k,v in self.templates:
            tempMat = getDistMat(testItem, v)
            newmindist =  DTWminDist(tempMat,0,0,0) 
            
            if self.minDist > newmindist:
                self.minDist = newmindist
                self.model = k

        print self.model
        
        return self.model 
