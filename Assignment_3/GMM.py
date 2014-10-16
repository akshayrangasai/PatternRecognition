import numpy as np
import math
from sklearn.cluster import KMeans

class GMM(object):

	def __init__(self, trainData, n_clusters, covar_type = "full"):
		#Initialize values using K-Means
		
		self.n_clusters = n_clusters
		self.covar_type = covar_type

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
			self.model_priors.append(float(len(cluster)/len(trainData)))

	def pdf(self, data):
		#Calculate probability of data in each component and return
		dim = len(data)
		prob = []
		for i in range(n_clusters):
			k = 1.0/(math.pow((2*pi), float(dim)/2) * math.pow(np.linalg.det(self.model_covar[i])))
			x_mu = matrix(data - self.model_centers[i])
			p = self.model_priors[i] * k * math.pow(math.e, -0.5 * (x_mu * self.model_covar.I * x_mu.T))
			prob.append(p)
		return prob

	def EMfit(self, trainData, n_iter):
		#Expectation Maximisation to fit GMM 
		(N,dim) = np.shape(trainData)
		data = np.matrix(trainData)
	
		for ii in range(n_iter)			
		
			#E step
			gamma = []
			for x in trainData: 
				gamma.append(pdf(self,x)/np.sum(pdf(self,x))) 

			#M step
			Gamma = np.array(gamma)
			Nk = np.sum(Gamma,axis=0)
			for i in range(n_clusters):
				mu = (1.0/Nk[i]) * np.dot(Gamma[:,i].T,data)
				sigma = zeros(dim,dim)

				for j in range(N):
					sigma += Gamma[j,i] * np.outer(data[j,:] - mu, data[j,:] - mu)

				sigma = sigma / Nk[i]

				self.model_centers[i] = mu
				if self.covar_type is "full":
					self.model_covar[i] = sigma
				elif self.covar_type is "diagonal":
					self.model_covar[i] = np.diag(np.diag(sigma))
				self.model_priors[i] = Nk[i] / np.sum(Nk)				
