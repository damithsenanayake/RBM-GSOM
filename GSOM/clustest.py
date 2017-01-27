import numpy as np
import matplotlib.pyplot as plt
from MovingMap import  MovingMap
from SelfOrganizingSwarm import SelfOrganizingSwarm
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from bgsom import GSOM
from sklearn.decomposition import PCA
from sklearn import datasets

pack = datasets.load_digits()
#datasets.load_iris()
D = pack.data#[:1000]
c = pack.target#[:1000]
Reducer = GSOM()#MovingMap(iterations=100, beta=0.5)


Y = Reducer.fit_transform(D, lr=1, beta=0.75, sf=0.1, fd = 0.1)
plt.subplot(211)
plt.scatter(Y.T[0], Y.T[1], s=15,alpha=0.75, edgecolors='none',  c = plt.cm.plasma(c.astype(float)/len(np.unique(c))))
plt.subplot(212)
plt.scatter(Y.T[0], Y.T[1], s= 15, alpha=0.75, edgecolors='none',  c = 'gray')

plt.show()