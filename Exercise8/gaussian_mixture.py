import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
# ============
# Generate datasets
# ============
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
aniso = (np.dot(X, transformation), y)
# The datasets, shown as rows
datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (no_structure, {})]

# ============
# Create cluster objects
# ============
two_means = cluster.MiniBatchKMeans(n_clusters=3)
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full')
# the algorithms shown as colors
clustering_algorithms = (
    ('MiniBatchKMeans', two_means), ('GaussianMixture', gmm)
)

# ============
# Set up figure settings
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plot_num = 1
plt.subplots_adjust(left=.02, right=.98, bottom=.001,
                    top=.96, wspace=.05, hspace=.01)
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # split in data and classes
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)
        y_pred = algorithm.predict(X)
        # make plot fancy
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], s=10, c=y_pred)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
plt.show()
