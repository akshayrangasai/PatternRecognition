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
        self.classclust = np.zeros(n_clusters)

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
        dim = np.shape(data)[0]
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
            old_ll = ll
            ll = 0
            for x in trainData:
                ll += np.log(np.sum(self.pdf(x)))
            self.loglik.append(ll)

            #if old_ll > ll:
            #    break

    def plotloglikelihood(self):
        plt.plot(self.loglik)
        plt.xlabel('Number of iterations')
        plt.ylabel('Log Likelihood')
        plt.show()

    def saveloglikelihood(self, filename):
        plt.plot(self.loglik)
        plt.xlabel('Number of iterations')
        plt.ylabel('Log Likelihood')
        plt.savefig(filename+'.png')

    def classcluster(self, classData, class_no):
        #call separately with data of all classes
        y = []
        for x in classData:
            y.append(self.pdf(x)/np.sum(self.pdf(x)))

        totes = np.sum(y,axis=0)
        print totes
        idx = totes.tolist().index(np.max(totes))
        self.classclust[idx] = class_no

        print self.classclust

    def predict_class(self,testData):
        y = []
        
        for x in testData:
            y.append(self.pdf(x)/np.sum(self.pdf(x)))

        p = np.zeros((np.shape(y)))

        for idx in range(self.n_clusters):
            p[:,self.classcluster[idx]] = np.array(y)[:,idx] 

        return np.array(p)

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
m = 1
trainingset = [[] for i in range(m)]
classdata = []
for k, v in trdict.iteritems():
    classset = [[] for i in range(m)]
    for _v in v:
        data = np.genfromtxt(rootpath+'/'+dirs[k]+'/'+_v)
        for i in range(m):
            trainingset[i].append(data[i].tolist())
            classset[i].append(data[i].tolist())
    classdata.append(np.array(classset))

GMMs = []
for i in range(m):
    print "GMM #", i
    GMMs.append(GMM(trainingset[i], len(dirs), "full"))
    GMMs[i].EMfit(trainingset[i], 2)
    #GMMs[i].saveloglikelihood('likelihood'+str(i))
    for j in range(len(dirs)):
        GMMs[i].classcluster(classdata[j][i],j)
    print GMMs[i].classclust

testset = [[] for i in range(m)]
#testtruth = [[] for i in range(m)]
#for k, v in testdict.iteritems():
for _v in testdict[1]:
    print _v
    #elementlist.append(list(chain.from_iterable(np.genfromtxt(rootpath +'/' + dirs[k] +'/' +_v).tolist())))
    data = np.genfromtxt(rootpath+'/'+dirs[1]+'/'+_v)
    for i in range(m):
        testset[i].append(data[i].tolist())
        #testtruth[i].append(k)

pred = np.zeros((np.shape(testset)[1],3))
for i in range(m):
    p = GMMs[i].predict_class(testset[i])
    pred += [a/np.sum(a) for a in p]
    print "GMM probability", [a/np.sum(a) for a in p]
    
print "Sum", pred


