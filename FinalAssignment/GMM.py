import numpy as np
import math,os
from sklearn.cluster import KMeans
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import mixture

# Read files - split to train and test
rootpath = 'datasets/Images/'
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
(m,n) = np.shape(datasample)

classdata = []
for k, v in trdict.iteritems():
    classset = []
    for _v in v:
        data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
        classset.append(data)
    classdata.append(np.concatenate(classset))
print np.shape(classdata[0]), np.shape(classdata[1]), np.shape(classdata[7])

#Train the GMMs for each class
GMMs = []
for i in range(len(dirs)):
    print "Training GMM for", dirs[i]
    GMMs.append(mixture.GMM(n_components=4))
    GMMs[i].fit(classdata[i])
            
#Use GMMs for testing  
confmat = np.zeros((len(dirs),len(dirs)))          
for k, v in testdict.iteritems(): 
    print 'Testing class', dirs[k]
    prediction = []
    for _v in v:
        posterior = []
        data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
        for i in range(len(dirs)):
            posterior.append(np.sum(GMMs[i].score(data)))
        prediction.append(posterior.index(np.max(posterior)))
    confmat[k] = [prediction.count(i) for i in range(len(dirs))]

Precision = []
Recall = []
for i in range(len(dirs)):
    Ncp = confmat[i,i]
    Nfp = np.sum(confmat[i,:]) - Ncp
    Nfn = np.sum(confmat[:,i]) - Ncp
    Precision.append(float(Ncp)/float(Ncp + Nfp))
    Recall.append(float(Ncp)/float(Ncp + Nfn))

print 'Precision', Precision
print 'Recall', Recall
print 'Confusion', confmat

plt.clf()
plt.matshow(confmat)
plt.colorbar()
plt.title('Confusion Matrix for the image dataset')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_GMM_image.png')
