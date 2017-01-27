from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from TSOS import SelfOrganizingSwarm
from MovingMap import MovingMap
from bgsom import GSOM


X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)
# Y = SelfOrganizingSwarm(iterations=250, alpha=1, beta = 0.9,delta=0.001, theta=3).fit_transform(X)
# Y = MovingMap(iterations=10,beta= 0 ).fit_transform(X)
Y = GSOM().fit_transform(X, beta=0.1)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X.T[0], X.T[1], X.T[2],c = color, alpha=0.5, edgecolors='none')
# plt.show()

# ax = fig.add_subplot(211)
plt.scatter(Y.T[0], Y.T[1], s = 15, c = color, edgecolors='none', alpha=0.5)

#
# Y = Isomap().fit_transform(X)
# ax2 = fig.add_subplot(121)
# ax2.scatter(Y.T[0], Y.T[1], c = color, edgecolors='none', alpha=0.5)

plt.show()