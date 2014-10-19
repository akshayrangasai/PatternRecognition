import numpy as np
import math,os
from sklearn.cluster import KMeans
from itertools import chain
import matplotlib.pyplot as plt

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
            if covar_type is "full":
                self.model_covar.append(np.cov(cluster, rowvar=0))
            elif covar_type is "diagonal":
                self.model_covar.append(np.diag(np.diag(np.cov(cluster, rowvar=0))))
            self.model_priors.append(float(len(cluster))/float(len(trainData)))

    def pdf(self, data):
        #print data.shape
        dim = data.shape[0]
        #Calculate probability of data in each component and return
        prob = []
        for i in range(self.n_clusters):
            x_mu = np.matrix(data - self.model_centers[i])
            covar = np.matrix(self.model_covar[i])
            A = np.linalg.inv(covar)
            det = np.fabs(np.linalg.det(covar))
            k = (2.0*np.pi)**(dim/2.0) * (det)**(0.5)
            p = self.model_priors[i] * np.exp(float(-0.5 * x_mu * A * x_mu.T)) / k
            if p == 0:
                p = np.exp(-745)
            prob.append(p)

        return prob

    def EMfit(self, trainData, n_iter = 100):
        #Expectation Maximisation to fit GMM 
        (N,dim) = np.shape(trainData)
        data = trainData
    
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
            for i in range(self.n_clusters):
                mu = (1.0/Nk[i]) * np.dot(Gamma[:,i].T,data)
                sigma = np.zeros((dim,dim))

                for j in range(N):
                    sigma += Gamma[j,i] * np.outer(data[j,:] - mu, data[j,:] - mu)
                #print Gamma[j,i]
                                
                sigma = sigma / Nk[i]

                self.model_centers[i] = mu
                if self.covar_type is "full":
                    self.model_covar[i] = sigma
                elif self.covar_type is "diagonal":
                    self.model_covar[i] = np.diag(np.diag(sigma))
                self.model_priors[i] = Nk[i] / np.sum(Nk)
            #print "Centeres", self.model_centers
            #print "Covariance", self.model_covar
            #print "Prior", self.model_priors
            #Log Likelihood
            ll = 0
            for x in trainData:
                ll += np.log(np.sum(self.pdf(x)))
            self.loglik.append(ll)

    def loglikelihood(self):
        return self.loglik


#data = []
#for i in range(0,20):
#    data.append([i, math.pow(i,0.5)+7,math.pow(i,2)+3])
#for i in range(60,80):
#    data.append([i,math.pow(i,0.5)+7, math.pow(i,2)+3])
#GMMtest = GMM(data,3,"full")
#GMMtest.EMfit(data,100)
rootpath = 'GMM/features'
path, dirs, files  = os.walk(rootpath).next()
datadict = dict()
labels = enumerate(dirs)
for di in range(0,len(dirs)):
    f = os.listdir(rootpath+ '/'+ dirs[di])
    datadict[di] = f

elementlist = []
for k, v in datadict.iteritems():
    for _v in v:
        #print _v
        #elementlist.append(list(chain.from_iterable(np.genfromtxt(rootpath +'/' + dirs[k] +'/' +_v).tolist())))
        elementlist.append(np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v))
trainingset = np.concatenate(elementlist)
print trainingset.shape
GMMtest = GMM(trainingset ,3,"full")
GMMtest.EMfit(trainingset, 20)
print GMMtest.loglikelihood()
plt.plot(GMMtest.loglikelihood())
plt.xlabel('Number of iterations')
plt.ylabel('Log Likelihood')
plt.show()