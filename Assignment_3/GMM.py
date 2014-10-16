import numpy as np
import random
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
			self.model_priors.append(float(len(trainData)/len(cluster)))

	def EMfit(self, trainData, n_iter):
		#Expectation Maximisation to fit GMM 

		for i in range(n_iter)
			
			#E step
			gamma = np.zeros(len(n_clusters),len(trainData))
			
			for x in trainData:
				for i in keys:
					gauss = multivariate_normal(self.centroids[i],self.covar[i])
					p = mc[i] * (2* np.pi)^(-) *  
					gamma[i,j] = 

			#M step
			for i in keys:
				for x in trainData:
					

				
