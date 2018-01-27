# coding: utf-8


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
# from scipy.cluster.hierarchy import set_link_color_palette
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# CellStrat - Clustering
#
# Source Credit : *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
#

# # Python Machine Learning

# # Working with Unlabeled Data – Clustering Analysis


# ### Overview

# - [Grouping objects by similarity using k-means](#Grouping-objects-by-similarity-using-k-means)
#   - [K-means clustering using scikit-learn](#K-means-clustering-using-scikit-learn)
#   - [A smarter way of placing the initial cluster centroids using k-means++](#A-smarter-way-of-placing-the-initial-cluster-centroids-using-k-means++)
#   - [Hard versus soft clustering](#Hard-versus-soft-clustering)
#   - [Using the elbow method to find the optimal number of clusters](#Using-the-elbow-method-to-find-the-optimal-number-of-clusters)
#   - [Quantifying the quality of clustering via silhouette plots](#Quantifying-the-quality-of-clustering-via-silhouette-plots)
# - [Organizing clusters as a hierarchical tree](#Organizing-clusters-as-a-hierarchical-tree)
#   - [Grouping clusters in bottom-up fashion](#Grouping-clusters-in-bottom-up-fashion)
#   - [Performing hierarchical clustering on a distance matrix](#Performing-hierarchical-clustering-on-a-distance-matrix)
#   - [Attaching dendrograms to a heat map](#Attaching-dendrograms-to-a-heat-map)
#   - [Applying agglomerative clustering via scikit-learn](#Applying-agglomerative-clustering-via-scikit-learn)
# - [Locating regions of high density via DBSCAN](#Locating-regions-of-high-density-via-DBSCAN)
# - [Summary](#Summary)






# # Grouping objects by similarity using k-means

# ## K-means clustering using scikit-learn




X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)





plt.scatter(X[:, 0], X[:, 1], 
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
#plt.savefig('images/11_01.png', dpi=300)
plt.title("some random data")
plt.show()





km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)




plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
#plt.savefig('images/11_02.png', dpi=300)
plt.title("detect centroids with K-means")
plt.show()



# ## Using the elbow method to find the optimal number of clusters
# to check quality of clustering, one has to use intrinsic methods, such as the within-cluster SSE (sum of square errors) also called Distortion.
# This is available via the inertia_ attribute after fitting a Kmeans model.



print('Distortion: %.2f' % km.inertia_)




distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
#plt.savefig('images/11_03.png', dpi=300)
plt.title("check optimum no of clusters with Elbow method")
plt.show()