import numpy as np
import os
from sklearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from sklearn.hmm import GaussianHMM as HMM
#from hmm.continuous.GMHMM import GMHMM as HMM
#from hmm.discrete.DiscreteHMM import DiscreteHMM as HMM
#model = GaussianHMM(n_components = 2)
rootpath = 'GMM/features'
path, dirs, files  = os.walk(rootpath).next()
datadict = dict()
trdict, testdict = dict(), dict()
labels = enumerate(dirs)
for di in range(0,len(dirs)):
    f = os.listdir(rootpath+ '/'+ dirs[di])
    datadict[di] = f

for di in range(0,len(dirs)):
    np.random.shuffle(datadict[di])
    tr_idx = 0.8*np.shape(datadict[di])[0]
    trdict[di], testdict[di] = datadict[di][:int(tr_idx)], datadict[di][int(tr_idx):]

datasample = np.genfromtxt(rootpath+'/'+dirs[0]+'/'+datadict[0][0])
img = imread('GMM/coast/coast_bea5.jpg')
#kmm = KMeans()
#kmm.fit(img)

'''
To solve this problem, we have to quantize the vectors. There are multiple ways of doing this, we can either do this by kmeans compression, and cluster the image and predict cluster numbers or visual inspection. 
'''

def quantize_data(datapoint, kmeansarray):
    quant = []
    for i in xrange(datapoint.shape[1]):
        quant.append(kmeansarray[i].predict(datapoint[:,i]))

    return np.asarray(quant)

split = np.shape(datasample)[0]
quantized_set = []
trainingset = []
testingset = []
classdata = []
testclassdata = []
classlabels = []
testclasslabels = []
kmms = []
for k, v in trdict.iteritems():
    classset = []
    testclassset = []
    i = 0
    for _v in v:
        if (i-k*len(v)/3.0) > 0 and  (i-k*len(v)/3.0) < 0.7*(len(v)/3.0):
            #print _v
            data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
            classlabels.append(k)#for i in range(m)
            trainingset.append(data)
            classset.append(data)
        else:
            data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
            testclasslabels.append(k)#for i in range(m)
            testingset.append(data)
            testclassset.append(data)
        i = i+1

    classdata.append(np.concatenate(classset))
    testclassdata.append(np.concatenate(testclassset))
trainingset = np.concatenate(trainingset)
testingset = np.concatenate(testingset)
print np.shape(trainingset)
for i in range(0,trainingset.shape[1]):
    trainset = trainingset[:,i].reshape(trainingset.shape[0]/split,split)
    kmm = KMeans()
    kmm.fit(trainset)
    kmms.append(kmm)
    quantized_set.append(kmm.labels_)
#print np.shape(classdata[0]), np.shape(classdata[1]), np.shape(classdata[2])
#print np.shape(quantized_set)
#print classlabels
quantized_set =  np.asarray(quantized_set)
nclasses = len(np.unique(classlabels))
hmmclass = []
#print classlabels
print quantized_set.shape
for i in range(0,nclasses):
    newtrainset = []
    for k in range(0,len(classlabels)):
        if classlabels[k] == i:
            #print i
            #print k
            newtrainset.append(quantized_set[:,k])
    
    newtrainset = np.asarray(newtrainset)
    print newtrainset.shape
    hmm = HMM(3)
    hmm.fit([newtrainset])
    hmmclass.append(hmm)

print testingset.shape
t = quantize_data(testingset[0:36,:], kmms)
print t.shape
for hm in hmmclass:
    print hm.score(t.T)
print testclasslabels[0]
