from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from TSOS import SelfOrganizingSwarm

X, color = datasets.samples_generator.make_s_curve(n_samples=1000)
Y = SelfOrganizingSwarm(iterations=250, alpha=1, beta = 0.9,delta=0.001, theta=3).fit_transform(X)
fig = plt.figure()

# try:
#     ax = fig.add_subplot(211, projection='3d')
#     ax.scatter(X.T[0],X.T[1], X.T[2], c= color, cmap=plt.cm.Spectral, alpha=0.5 )
# except:
#     ax = fig.add_subplot(211)
#     ax.scatter(X.T[0], X.T[2], c = color, cmap=plt.cm.Spectral, alpha=0.5)
# plt.show()

# ax = fig.add_subplot(211)
plt.scatter(Y.T[0], Y.T[1], c = color, edgecolors='none', alpha=0.5)

#
# Y = Isomap().fit_transform(X)
# ax2 = fig.add_subplot(121)
# ax2.scatter(Y.T[0], Y.T[1], c = color, edgecolors='none', alpha=0.5)

plt.show()