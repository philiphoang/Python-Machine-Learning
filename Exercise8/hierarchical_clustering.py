import scipy.cluster.hierarchy as hier
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import numpy as np
import distance  # <- you most likely need pip install distance here

data = ["ATGTAAA", "ATGAAAA", "ACGTGAA", "ACGAGGG", "ACGAGGA", "ACGAGTC", "ACGAGCC"]
labels = ["shark", "ray-finned fish", "amphibians", "primates", "rodents", "crocodiles", "dinosaurs"]

# The levenshtein distance matrix ->
mat = np.zeros((len(data), len(data)), dtype=int)
for i in range(0, len(data)):
    for j in range(0, len(data)):
        mat[i][j] = distance.levenshtein(data[i], data[j])
        print(mat[i][j], end="")
    print("\n")
mat = pdist(mat)  # make an upper triangle matrix
# The hierarchy clustering ->
# here run scipy.cluster.hierarchy.linkage() on triangle matrix
z = hier.linkage(mat)
fig = plt.figure(figsize=(25, 10))
# here run scipy.cluster.hierarchy.dendrogram() with the linkage z and labels = labels
dn = hier.dendrogram(z, labels = labels)
plt.show()
