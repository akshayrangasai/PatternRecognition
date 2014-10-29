import numpy as np 
import matplotlib.pyplot as plt

dirs = ['eight', 'seven', 'six', 'three', 'two']
confmat = np.zeros((5,5))        
for k in range(5):
	prediction = [] 
	posterior = [[] for i in range(32)]
	for i in range(32):
		for y in dirs:
			data = np.genfromtxt(dirs[k]+'/'+y+'.txt')     
			posterior[i].append(data[i]) 
	for a in posterior:
		prediction.append(a.index(np.max(a)))
	confmat[k] = [prediction.count(i) for i in range(5)]

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

plt.matshow(confmat)
plt.colorbar()
plt.title('Confusion Matrix for the image dataset')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_digit_HMM.png')