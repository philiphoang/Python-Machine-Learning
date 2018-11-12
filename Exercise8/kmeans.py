import numpy as np; import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from lloyds_algorithm import kmeans

np.random.seed(0)
# ============
# Generate datasets
# ============
n_samples = 1500
X, y = datasets.make_blobs(n_samples=n_samples, random_state=100)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X, y = (np.dot(X, transformation), y)

# your method ->
centers = kmeans(X, 3)
colors = [] # colors for plot
new_centers = [] # reshape, the old way
for i in range(len(centers)):
    colors.extend([i] * len(centers[i]))
    new_centers.extend(centers[i])

# sklearn version
centersSKL = cluster.MiniBatchKMeans(n_clusters=3)
centersSKL.fit(X)
y_pred = centersSKL.predict(X)
# ============
# Set up figure settings
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
x = [ x[0] for x in new_centers]
y = [ x[1] for x in new_centers]

plt.subplot(1, 2, 1) # your subplot
plt.title("My implementation")
plt.scatter(x, y, c=colors)
plt.xticks(());plt.yticks(())
plt.subplot(1, 2, 2) # sklearn subplot
plt.title("sklearn's implementation")
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.xticks(());plt.yticks(())
plt.show()