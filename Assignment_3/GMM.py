import numpy as np
import math,os
from sklearn.cluster import KMeans
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class GMM(object):

    def __init__(self, trainData, n_clusters, covar_type = "full"):
        #Initialize values using K-Means
        
        (N,dim) = np.shape(trainData)
        self.n_clusters = n_clusters
        self.covar_type = covar_type
        self.model_centers = []
        self.model_covar = []
        self.model_priors = []
        self.loglik = []

        kmeans = KMeans(init = 'k-means++', n_clusters = n_clusters, n_jobs = -1)
    
        label = kmeans.fit_predict(trainData)
        clusters = [[] for i in range(n_clusters)]
        for (l,d) in zip(label,trainData):
            clusters[l].append(d)

        for cluster in clusters:
            self.model_centers.append(np.mean(cluster, axis=0))
            if dim == 1:
                self.model_covar.append(np.var(cluster))
            else:
                if covar_type is "full":
                    self.model_covar.append(np.cov(cluster, rowvar=0))
                elif covar_type is "diagonal":
                    self.model_covar.append(np.diag(np.diag(np.cov(cluster, rowvar=0))))
            self.model_priors.append(1.0/n_clusters)

    def pdf(self, data):
        #print data.shape
        dim = np.shape(data)[0]
        #Calculate probability of data in each component and return
        prob = []
        for i in range(self.n_clusters):
            x_mu = np.matrix(data - self.model_centers[i])
            covar = np.matrix(self.model_covar[i])
            if dim == 1:
                A = 1.0/covar
                det = np.fabs(covar[0])
            else:
                A = np.linalg.inv(covar)
                det = np.fabs(np.linalg.det(covar))
            k = (2.0*np.pi)**(dim/2.0)  * np.array(det)**(0.5)
            p = self.model_priors[i] * np.exp(float(-0.5 * x_mu * A * x_mu.T)) / k
            if p == 0:
                p = np.exp(-745)
            prob.append(p)

        #print prob
        return prob

    def EMfit(self, trainData, n_iter = 100):
        #Expectation Maximisation to fit GMM 
        (N,dim) = np.shape(trainData)
        data = np.array(trainData)
        ll = np.exp(-700) #large negative initial value for init
    
        for k in range(n_iter):         
    
            #E step
            print "Iter ", k
            gamma = []
            for x in trainData:
                gamma.append(self.pdf(x)/np.sum(self.pdf(x)))
                
            #M step
            Gamma = np.array(gamma)
            #print np.shape(Gamma)
            Nk = np.sum(Gamma,axis=0)
            self.model_centers = []
            for i in range(self.n_clusters):
                mu = (1.0/Nk[i]) * np.squeeze(np.dot(Gamma[:,i].T,data))
                sigma = np.zeros((dim,dim))

                for j in range(N):
                    sigma += Gamma[j,i] * np.outer(data[j,:] - mu, data[j,:] - mu)
                #print Gamma[j,i]
                                
                sigma = sigma / Nk[i]

                self.model_centers.append(mu)
                if self.covar_type is "full":
                    self.model_covar[i] = sigma
                elif self.covar_type is "diagonal":
                    self.model_covar[i] = np.diag(np.diag(sigma))
                self.model_priors[i] = Nk[i] / np.sum(Nk)

            #print "Centeres", self.model_centers
            #print "Covariance", self.model_covar
            #print "Prior", self.model_priors
            #Log Likelihood
            old_ll = ll
            ll = 0

            for x in trainData:
                ll += np.log(np.sum(self.pdf(x)))
            self.loglik.append(ll)

            #if ll - old_ll < 0.01:
            #    print "Converged!"
            #    return

    def plotloglikelihood(self):
        plt.clf()
        plt.plot(self.loglik)
        plt.xlabel('Number of iterations')
        plt.ylabel('Log Likelihood')
        plt.show()

    def saveloglikelihood(self, filename):
        plt.clf()
        plt.plot(self.loglik)
        plt.xlabel('Number of iterations')
        plt.ylabel('Log Likelihood')
        plt.savefig(filename+'.png')

    def predict(self,testData):
        p = 0
        for x in testData:
            p += np.log(np.sum(self.pdf(x)))

        return p

def putplots(k,clusters,iters):
        #plot points
    colors = ["r","b","g"]
    #for i in range(3):
    #    plt.scatter(classdata[i][:,0],classdata[i][:,1], s=1, color = colors[i], marker = 'o')
    #plt.savefig('trainingdata_scatter.png')

    plt.clf()
    x = np.arange(0.0,1.0,0.01)
    y = np.arange(0.0,1.0,0.01)
    X,Y = np.meshgrid(x,y)
    z = []
    for i in range(clusters):
        sigma_x = math.sqrt(GMMs[k].model_covar[i][0,0])
        sigma_y = math.sqrt(GMMs[k].model_covar[i][1,1])
        Z = mlab.bivariate_normal(X,Y,sigma_x,sigma_y, GMMs[k].model_centers[i][0], GMMs[k].model_centers[i][1],GMMs[k].model_covar[i][1,0])
        Cp = plt.contour(X,Y,Z)
        plt.clabel(Cp, inline = 1, fontsize = 10)

    plt.scatter(classdata[k][:,0],classdata[k][:,1], s=1, color = colors[k%3], marker = 'o')
    plt.title('Training Data and mixture components')
    plt.savefig('contours_diag'+dirs[k]+str(iters)+'.png')

# Read files - split to train and test
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
(m,n) = np.shape(datasample)

classdata = []
for k, v in trdict.iteritems():
    classset = []
    for _v in v:
        data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
        classset.append(data)
    classdata.append(np.concatenate(classset))
print np.shape(classdata[0]), np.shape(classdata[1]), np.shape(classdata[2])

#Train the GMMs for each class
iters = [0,5,15,30]
for n_iter in iters:
    GMMs = []
    for i in range(len(dirs)):
        print "Training GMM for", dirs[i]
        GMMs.append(GMM(classdata[i], 4, "diagonal"))
        GMMs[i].EMfit(classdata[i], n_iter)
        putplots(i,4,n_iter)
        #GMMs[i].saveloglikelihood('likelihood_digits'+str(i))
            
#Use GMMs for testing  
confmat = np.zeros((len(dirs),len(dirs)))          
for k, v in testdict.iteritems(): 
    print 'Testing class', dirs[k]
    prediction = []
    for _v in v:
        posterior = []
        data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
        for i in range(len(dirs)):
            posterior.append(GMMs[i].predict(data))
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
#plt.savefig('confusion_digit.png')
