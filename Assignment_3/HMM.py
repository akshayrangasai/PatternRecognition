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
rootpath = 'Digits/digit_data'#'GMM/features'
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
print split
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
        if i < 0.7*len(v):
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
#print np.shape(trainingset)
for i in range(0,trainingset.shape[1]):
    #print trainingset[:,i].shape[0]/split, split
    print split
    trainset = trainingset[:,i]
    trainset = trainset.reshape((trainingset.shape[0]/split,split))
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
    #print newtrainset.shape
    hmm = HMM(64)
    hmm.fit([newtrainset])
    hmmclass.append(hmm)

#print testingset.shape
rowdivision = datasample.shape[0]
t = []
for i in xrange(int(round(testingset.shape[0]/rowdivision))):
    t.append( quantize_data(testingset[rowdivision*i:rowdivision*(i+1),:], kmms))
#print t.shape
t = np.asarray(t)
rlabels = []
for ts in t:
    i = 0
    index = 0
    minlikely = -10000000
    for hm in hmmclass:
        sc =  hm.score(ts.T)
        if sc >  minlikely:
            index = i
            minlikely = sc
        i = i+1
    rlabels.append(index)

confmat = np.zeros((nclasses,nclasses))
print len(rlabels), len(testclasslabels)

for t in range(0,len(rlabels)):
    #print t
    confmat[testclasslabels[t],rlabels[t]] = confmat[testclasslabels[t],rlabels[t]]  + 1
plt.matshow(confmat)
plt.colorbar()
plt.title('Confusion Matrix for the image dataset')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

