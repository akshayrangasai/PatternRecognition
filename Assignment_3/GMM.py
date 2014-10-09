import numpy as np
import random

class GMM:

	def __init__(self):
		self.model = None
		self.nclusters = None

	def initclusters(self, trainData, n_clusters):
		#use k-means to estimate intial cluster centres
		oldmu = random.sample(trainData,n_clusters)
		mu = random.sample(trainData,n_clusters)
		converge = False

		while converge:
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

	def EMfit(self, trainData, mu, clusters):
		while 
