import numpy as np
import random

class GMM:

	def __init__(self):
		self.centroids = None
		self.covar = None
		self.mc = None
		self.nclusters = None

	def initclusters(self, trainData, n_clusters):
		#use k-means to estimate intial cluster centres
		oldmu = random.sample(trainData,n_clusters)
		mu = random.sample(trainData,n_clusters)
		converge = False

		while not converge:
			oldmu = mu
			#cluster points
			for x in trainData:
				clusters = {}
				min_dist = np.linalg.norm(x-mu[0])
				min_key = 0
				for i in enumerate(mu):
					dist = np.linalg.norm(x-mu[i[0]])
					if dist < min_dist:
						min_dist = dist
						min_key = i[0]
				try:
					clusters[min_key].append(x)
				except KeyError:
					clusters[min_key] = [x]
			#calculate new centres
			keys = sorted(clusters.keys())
			mu = []
			for i in keys:
				mu.append(np.mean(clusters[i], axis = 0))
			#check for convergence
			converge = set(tuple(a) for a in mu) == set(tuple(a) for a in oldmu)
		return(mu, clusters) 		

	def fitGMM(self, trainData, mu, clusters, cov_type):
		#Expectation Maximisation to fit GMM 
		self.covar = []
		mc = []
		keys = sorted(clusters.keys())
		
		self.centroids = mu

		if cov_type == 1: #full covariance matrix
			for i in keys:
				self.covar.append(np.cov(clusters[i],rowvar=0))
		
		else if cov_type == 2: #diagonal covariance matrix
			for i in keys:
				self.covar.append(np.cov(clusters[i], rowvar=0))
			np.diag(np.diag(self.covar))

		for i in keys:
			mc.append(len(clusters[i]))
		self.mc = [a / sum(mc) for a in mc]

		converge = False


		while not converge:
			#E step
			gamma = {}
			










