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
    x = np.arange(-20.0,20.0,0.1)
    y = np.arange(-20.0,20.0,0.1)
    X,Y = np.meshgrid(x,y)
    z = []
    for i in range(clusters):
        sigma_x = math.sqrt(GMMs[k].model_covar[i][0,0])
        sigma_y = math.sqrt(GMMs[k].model_covar[i][1,1])
        #print GMMs[k].model_covar, GMMs[k].model_centers
        #print sigma_x, sigma_y
        #print 'rho', GMMs[k].model_covar[i][1,0]/(sigma_x*sigma_y)
        Z = mlab.bivariate_normal(X,Y,sigma_x,sigma_y, GMMs[k].model_centers[i][0], GMMs[k].model_centers[i][1],GMMs[k].model_covar[i][1,0])
        #print Z
        Cp = plt.contour(X,Y,Z)
        plt.clabel(Cp, inline = 1, fontsize = 10)

    plt.scatter(classdata[k][:,0],classdata[k][:,1], s=1, color = colors[k], marker = 'o')
    plt.title('Training Data and mixture components')
    plt.savefig('contours_spiral'+str(k)+str(iters)+'.png')

# Read files - split to train and test
dataset = np.genfromtxt('spiraldata.txt')
(m,n) = np.shape(dataset)

classdata = [dataset[:400,:2],dataset[500:900,:2]]
print np.shape(classdata[0]), np.shape(classdata[1])

testdata = [dataset[400:500,:2],dataset[900:,:2]]
print np.shape(testdata[0]), np.shape(testdata[1])

#Train the GMMs for each class
iters = [5]
for n_iter in iters:
    GMMs = []
    M = [15, 10]
    for i in range(2):
        print "Training GMM for", i
        GMMs.append(GMM(classdata[i], M[i], "full"))
        GMMs[i].EMfit(classdata[i], n_iter)
        putplots(i,M[i],n_iter)
        GMMs[i].saveloglikelihood('likelihood_spiral'+str(i))
            
#Use GMMs for testing  
confmat = np.zeros((2,2))          
for j in range(2):
    print 'Testing class', j
    prediction = []
    for data in testdata[j]:
        posterior = []
        for i in range(2):
            posterior.append(GMMs[i].predict([data]))
        prediction.append(posterior.index(np.max(posterior)))
    confmat[j] = [prediction.count(i) for i in range(2)]

Precision = []
Recall = []
for i in range(2):
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
plt.savefig('confusion_digit.png')
