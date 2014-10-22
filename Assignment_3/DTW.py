import numpy as np
import os    
import matplotlib.pyplot as plt
#def DTWminDist(mat, x,y,dist):
 

class DTW:
    
    def getDistMat(self, testItem, val):
       
        distMat = np.zeros(((testItem.shape[0]), val.shape[0]))
        
        distMat[0,0] = np.linalg.norm(testItem[0,:]- val[0,:])
        for i in range(0,distMat.shape[0]):
            for j in range(0,distMat.shape[1]):
                
                if i>1 and j > 0 :
                    distMat[i,j] = min(distMat[i,j-1], distMat[i-1,j-1], distMat[i-2,j-1]) + np.linalg.norm(testItem[i,:]- val[j,:])

                if i > 0 and i <= 1 and j>0:
                    distMat[i,j] = min(distMat[i,j-1], distMat[i-1,j-1]) + np.linalg.norm(testItem[i,:]- val[j,:])

                if j == 0 and i > 0:
                    distMat[i,j] =  np.linalg.norm(testItem[i,:]- val[j,:])

                if i ==0 and j > 0:
                    distMat[i,j] = distMat[i,j-1] + np.linalg.norm(testItem[i,:]- val[j,:])

        '''
        DTW for  sequences.
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
        
        '''
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
path, dirs, files  = os.walk('Digits/digit_data/eight').next()
totfiles = len(files)
print totfiles
DTWdistMat = np.zeros((1,totfiles))
for k in range(0, 1):
    for l in range(0,totfiles):
        test = np.genfromtxt('Digits/digit_data/eight/%d.txt'%(l+1))
        temp = np.genfromtxt('Digits/digit_data/eight/%d.txt'%(k+1))
        DTWdistMat[k,l] = dttest.getDistMat(temp,test)[temp.shape[0]-1,test.shape[0]-1] 
'''
test = np.genfromtxt('Digits/digit_data/eight/1.txt')
temp = np.genfromtxt('Digits/digit_data/eight/1.txt')
test2 = np.genfromtxt('Digits/digit_data/one/2.txt')

#print temp[0,:]
print dttest.getDistMat(temp,test)[temp.shape[0]-1,test.shape[0]-1], dttest.getDistMat(temp,test2)[temp.shape[0]-1,test2.shape[0]-1]
'''
histmat, bins = np.histogram((DTWdistMat),bins = 100)
center = (bins[:-1]- bins[1:])/2 
width = 0.7*(bins[1]-bins[0])
plt.bar(center,histmat,align = 'center', width = width)
#plt.hist(histmat)
plt.plot(DTWdistMat)
plt.show()
#print histmat
