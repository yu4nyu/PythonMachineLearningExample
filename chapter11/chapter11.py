### 001 create data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5,
        shuffle=True, random_state=0)

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c='', marker='o', s=50, facecolors='none', edgecolors='black')
plt.grid()
plt.show()



### 002
from sklearn.cluster import KMeans
# Set n_init=10 to run the k-means clustering algorithms 10 times independently with
# different random centroids to choose the final model as the one with the lowest SSE.
# tol is a parameter that controls the tolerance with regard to the changes in the
# within-cluster sum-squared-error to declare convergence.
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
