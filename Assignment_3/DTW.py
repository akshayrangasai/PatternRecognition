import numpy as np
    

#def DTWminDist(mat, x,y,dist):
 

class DTW:
    
    def getDistMat(self, testItem, val):
       
        distMat = np.zeros((len(testItem), len(val)))

        if testItem[0] !=  val[0]:
            distMat[0,0] = 1

        for i in range(0,len(testItem)):
            for j in range(0,len(val)):
                if j == 0 and i > 0:
                    distMat[i,j] = distMat[i-1,j] + 1

                if i == 0 and j > 0:
                    distMat[i,j] = distMat[i,j-1] + 1

                if i>0 and j > 0:
                    distMat[i,j] = min((distMat[i-1,j-1] + (testItem[i] != val[j])), distMat[i-1,j] + 1, distMat[i,j-1] +1 )
        
        
        return distMat

    def __init__(self, **kwargs):
        self.templates = kwargs.items()
        self.model = None
        self.minDist = 10e18
    
    def fit(self, testItem):
        
        for k,v in self.templates:
            tempMat = getDistMat(testItem, v)
            #newmindist =  DTWminDist(tempMat,0,0,0) 
            self.k = tempMat[len(testItem)-1, len(v) -1]
            #if self.minDist > newmindist:
            #    self.minDist = newmindist
            #    self.model = k

        print self.k
        
        return self.model





dttest = DTW()
test = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
temp = np.asarray([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
print dttest.getDistMat(test, temp)
