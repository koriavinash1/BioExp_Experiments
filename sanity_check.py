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
from sklearn.metrics import silhouette_score

class Cluster_Characteristics:

	def __init__(self, model, layer, weights = None):

		self.model = load_model(model, custom_objects={'gen_dice_loss':gen_dice_loss,
                                'dice_whole_metric':dice_whole_metric,
                                'dice_core_metric':dice_core_metric,
                                'dice_en_metric':dice_en_metric})
		self.layer = layer

		self.weights = self.model.get_layer(self.layer).get_weights()[0]


	def cluster_scores(self, position=False, kmax = 10):

		flatten_shape = self.weights.shape[0]*self.weights.shape[1]*self.weights.shape[2]
		X = self.weights.reshape((flatten_shape, -1))

		if position:
			position = np.repeat(np.array(list(range(flatten_shape)))[:, None], self.weights.shape[-1], axis = 1)
			print(position.shape)
			X = X + position

		X = X.transpose()

		sse = {'km':[], 'agg':[]}
		sil = {'km':[], 'agg':[]}

		for k in range(2, kmax+1):
			aggmodel = AgglomerativeClustering(n_clusters = k).fit(X)
			kmeans = KMeans(n_clusters = k).fit(X)
			centroids_km = kmeans.cluster_centers_
			pred_clusters_km = kmeans.predict(X)
			curr_sse_km = 0

			# calculate square of Euclidean distance of each point from its cluster center and add to current WSS
			for i in range(len(X)):
				curr_center_km = centroids_km[pred_clusters_km[i]]
				curr_sse_km += np.linalg.norm(X[i] - curr_center_km)**2

			sse['km'].append(curr_sse_km)

			# Calculate silhouette score
			labels_km = kmeans.labels_
			labels_agg = aggmodel.labels_
			sil['km'].append(silhouette_score(X, labels_km, metric = 'euclidean'))
			sil['agg'].append(silhouette_score(X, labels_agg, metric = 'euclidean'))

		return(sil, sse)

	def plot_dendrogram(self, X):

		plt.figure(figsize=(10, 7))  
		plt.title("Dendrograms")  
		dend = shc.dendrogram(shc.linkage(X, method='ward'))
		plt.show()


if __name__ == "__main__":

	C = Cluster_Characteristics('/home/parth/Interpretable_ML/saved_models/model_flair_scaled/model-archi.h5', 'conv2d_13',
		'/home/parth/Interpretable_ML/saved_models/model_flair_scaled/model-wts-flair.hdf5')

	sil, sse = C.cluster_scores(position = True)

	print(sil, sse)