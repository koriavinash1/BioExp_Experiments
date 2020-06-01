import numpy as np
from keras.models import load_model
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import sys
sys.setrecursionlimit(10**6) 
import matplotlib.pyplot as plt
sys.path.append('../BioExp')
from BioExp.helpers.metrics import *
from BioExp.helpers.losses import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples

class Cluster_Characteristics:

	def __init__(self, model, layer, weights = None):

		self.model = load_model(model, custom_objects={'gen_dice_loss':gen_dice_loss,
                                'dice_whole_metric':dice_whole_metric,
                                'dice_core_metric':dice_core_metric,
                                'dice_en_metric':dice_en_metric})

		self.model.load_weights(weights)

		self.layer = layer

		self.weights = self.model.get_layer(self.layer).get_weights()[0]

		self.kmax = None
		self.order = 1

	def cluster(self, position=False, kmax = 10, km = False, order=1):

		flatten_shape = self.weights.shape[0]*self.weights.shape[1]*self.weights.shape[2]
		X = self.weights.reshape((flatten_shape, -1))

		if position:
			position = np.repeat(np.array(list(range(flatten_shape)))[:, None], self.weights.shape[-1], axis = 1)
			X = X + position

		X = X.transpose()

		dist = self.distance_matrix(X,  order)

		self.scores = {'sil':[], 'cal':[], 'dav':[], 'wss':[]}

		for k in range(2, kmax+1):
			aggmodel = AgglomerativeClustering(n_clusters = k, affinity='precomputed', linkage='average')
			self.cluster_scores(aggmodel, dist, X)

		self.kmax = np.argmax(self.scores['sil'])

		self.order = order

	def cluster_scores(self, model, dist, X):

		labels = model.fit_predict(dist)

		self.scores['sil'].append(silhouette_score(dist, labels, metric = 'precomputed'))
		self.scores['cal'].append(calinski_harabasz_score(X, labels))
		self.scores['dav'].append(davies_bouldin_score(X, labels))

		# try:
		# 	kmeans = model.fit(X)
		# 	centroids_km = model.cluster_centers_
		# 	pred_clusters_km = model.predict(X)
		# 	curr_sse_km = 0

		# 	# calculate square of Euclidean distance of each point from its cluster center and add to current WSS
		# 	for i in range(len(X)):
		# 		curr_center_km = centroids_km[pred_clusters_km[i]]
		# 		curr_sse_km += np.linalg.norm(X[i] - curr_center_km)**2

		# 	self.scores['wss'].append(curr_sse_km)

		# 	# Calculate silhouette score
		# 	labels = model.labels_
		# 	self.scores['sil'].append(silhouette_score(X, labels_km, metric = 'euclidean'))

	def plot_dendrogram(self, X):

		plt.figure(figsize=(10, 7))  
		plt.title("Dendrograms")  
		dend = shc.dendrogram(shc.linkage(X, method='ward'))
		plt.show()

	def distance_matrix(self, X, f):

		distance_matrix = np.zeros((X.shape[0], X.shape[0]))

		for i in range(X.shape[0]):
			for j in range(X.shape[0]):
				distance_matrix[i][j] = np.linalg.norm(X[i] - X[j], ord = f)

		return(distance_matrix)

	def plot_silhouette(self, position=False, kmax = 10):

		flatten_shape = self.weights.shape[0]*self.weights.shape[1]*self.weights.shape[2]
		X = self.weights.reshape((flatten_shape, -1))

		if position:
			position = np.repeat(np.array(list(range(flatten_shape)))[:, None], self.weights.shape[-1], axis = 1)
			X = X + position

		X = X.transpose()

		dist = self.distance_matrix(X,  self.order)

		if self.kmax is not None:
			n_range = range(self.kmax+2, self.kmax+3)
			fig = plt.figure()
			fig.set_size_inches(10, 5)

			for n_clusters in n_range:
				plt.xlim([-0.1, 0.3])
				plt.ylim([0, len(X) + (n_clusters + 1) * 10])
				clusterer = AgglomerativeClustering(n_clusters = n_clusters, affinity='precomputed', linkage='average')
				cluster_labels = clusterer.fit_predict(dist)
				silhouette_avg = silhouette_score(dist, cluster_labels, metric = 'precomputed')
				print("For n_clusters =", n_clusters,
				  "The average silhouette_score is :", silhouette_avg)
				sample_silhouette_values = silhouette_samples(dist, cluster_labels, metric = 'precomputed')

				y_lower = 10
				for i in range(n_clusters):
					# Aggregate the silhouette scores for samples belonging to
					# cluster i, and sort them
					ith_cluster_silhouette_values = \
					    sample_silhouette_values[cluster_labels == i]

					ith_cluster_silhouette_values.sort()

					size_cluster_i = ith_cluster_silhouette_values.shape[0]
					y_upper = y_lower + size_cluster_i

					color = plt.cm.nipy_spectral(float(i) / n_clusters)
					plt.fill_betweenx(np.arange(y_lower, y_upper),
					                  0, ith_cluster_silhouette_values,
					                  facecolor=color, edgecolor=color, alpha=0.7)

					# Label the silhouette plots with their cluster numbers at the middle
					plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

					# Compute the new y_lower for next plot
					y_lower = y_upper + 10  # 10 for the 0 samples

					# ax[idx].set_title("The silhouette plot for the various clusters.")
					plt.xlabel("The silhouette coefficient values")
					plt.ylabel("Cluster label")

					# The vertical line for average silhouette score of all the values
					plt.axvline(x=silhouette_avg, color="red", linestyle="--")

					plt.yticks([])  # Clear the yaxis labels / ticks
					# plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

					plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
					          "with n_clusters = %d" % n_clusters),
					         fontsize=14, fontweight='bold')

			plt.show()
		else: 
			n_range = range(2, kmax+1)
			fig, ax = plt.subplots(len(list(n_range))+1)
			fig.set_size_inches(20, 60)

			for n_clusters in n_range:
				idx = n_clusters - list(n_range)[0]
				# The 1st subplot is the silhouette plot
				# The silhouette coefficient can range from -1, 1 but in this example all
				# lie within [-0.1, 1]
				ax[idx].set_xlim([-0.1, 1])
				# The (n_clusters+1)*10 is for inserting blank space between silhouette
				# plots of individual clusters, to demarcate them clearly.
				ax[idx].set_ylim([0, len(X) + (n_clusters + 1) * 10])

				# Initialize the clusterer with n_clusters value and a random generator
				# seed of 10 for reproducibility.
				clusterer = AgglomerativeClustering(n_clusters = n_clusters, affinity='precomputed', linkage='average')
				cluster_labels = clusterer.fit_predict(dist)

				# The silhouette_score gives the average value for all the samples.
				# This gives a perspective into the density and separation of the formed
				# clusters
				# The silhouette_score gives the average value for all the samples.
				# This gives a perspective into the density and separation of the formed
				# clusters
				silhouette_avg = silhouette_score(dist, cluster_labels, metric = 'precomputed')
				print("For n_clusters =", n_clusters,
				  "The average silhouette_score is :", silhouette_avg)

				# Compute the silhouette scores for each sample
				sample_silhouette_values = silhouette_samples(dist, cluster_labels, metric = 'precomputed')

				y_lower = 10
				for i in range(n_clusters):
					# Aggregate the silhouette scores for samples belonging to
					# cluster i, and sort them
					ith_cluster_silhouette_values = \
					    sample_silhouette_values[cluster_labels == i]

					ith_cluster_silhouette_values.sort()

					size_cluster_i = ith_cluster_silhouette_values.shape[0]
					y_upper = y_lower + size_cluster_i

					color = plt.cm.nipy_spectral(float(i) / n_clusters)
					ax[idx].fill_betweenx(np.arange(y_lower, y_upper),
					                  0, ith_cluster_silhouette_values,
					                  facecolor=color, edgecolor=color, alpha=0.7)

					# Label the silhouette plots with their cluster numbers at the middle
					ax[idx].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

					# Compute the new y_lower for next plot
					y_lower = y_upper + 10  # 10 for the 0 samples

					# ax[idx].set_title("The silhouette plot for the various clusters.")
					ax[idx].set_xlabel("The silhouette coefficient values")
					ax[idx].set_ylabel("Cluster label")

					# The vertical line for average silhouette score of all the values
					ax[idx].axvline(x=silhouette_avg, color="red", linestyle="--")

					ax[idx].set_yticks([])  # Clear the yaxis labels / ticks
					ax[idx].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

					plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
					          "with n_clusters = %d" % n_clusters),
					         fontsize=14, fontweight='bold')

			plt.show()

if __name__ == "__main__":

	C = Cluster_Characteristics('/home/parth/Interpretable_ML/saved_models/model_flair_scaled/model-archi.h5', 'conv2d_15',
		weights = '/home/parth/Interpretable_ML/saved_models/model_flair_scaled/model-wts-flair.hdf5')

	# sil, sse, cal, dav = C.cluster_scores(position = False)

	# print(sil, cal, dav)

	C.cluster(position=True, order = 0.1)

	C.plot_silhouette(position=True, kmax=10)