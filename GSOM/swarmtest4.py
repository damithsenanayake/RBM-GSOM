import numpy as np
import sklearn.datasets as ds
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


X, labels = ds.make_moons(n_samples=1000, noise= 0.01)

noise = np.random.randn(X.shape[0], 5)

# plt.scatter(X.T[0], X.T[1], c= y, alpha=0.5)
# plt.show()


X = np.append(X,noise, axis=1)
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.scatter(X.T[0], X.T[1], X.T[2],c = labels, alpha=0.5, edgecolors='none')
# plt.show()


Y = SelfOrganizingSwarm(iterations=50, alpha=1, beta=1, delta=0.0 , theta=3.5).fit_transform(X)
# Y = TSNE(2).fit_transform(X)
x, y = Y.T


plt.scatter(x,y, c= labels, alpha=0.1, edgecolors='none', s = 40)
plt.show()
