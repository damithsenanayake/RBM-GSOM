import numpy as np
import pandas as pd
from AEGSOM import  GSOM
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
data = pd.read_csv("~/data/5D3C_RAW.csv", header=None)

X = np.array(data)[:, :2]


# X = normalize(X, axis = 0)

# add noise

noise = np.random.randn(300, 90)*2
X = np.append(X,noise, axis = 1)

X += abs(X.min())
X /= X.max()

indices = np.random.permutation(X.shape[0])
gsom = GSOM(dims=92, hid = 2, sf = 0.8, fd = 0.9, max_nodes = 2000, min_nodes = 10, radius=60, X = X, gaussian=True, nei=True)

for i in range(40):
    print "\nbatch ", (i+1)
    gsom.train_batch(X[indices], iterations=1, lr = 0.25*np.exp(-i/10), prune=False)

# gsom.prune()


grid, hits = gsom.predict(X)
x, y = grid.T
# x, y = TSNE(2).fit_transform(X).T
plt.scatter(X.T[0], X.T[1])
plt.show()

plt.scatter(x[:100], y[:100], color = 'red',marker = 4)
plt.scatter(x[100:200], y[100:200], color = 'blue', marker = 5)
plt.scatter(x[200:], y[200:], color = 'green', marker = 6)
plt.show()
x, y = PCA(2).fit_transform(X).T
plt.scatter(x[:100], y[:100], color = 'red',marker = 4)
plt.scatter(x[100:200], y[100:200], color = 'blue', marker = 5)
plt.scatter(x[200:], y[200:], color = 'green', marker = 6)
plt.show()

x,y = TSNE(2).fit_transform(X).T
plt.scatter(x[:100], y[:100], color = 'red',marker = 4)
plt.scatter(x[100:200], y[100:200], color = 'blue', marker = 5)
plt.scatter(x[200:], y[200:], color = 'green', marker = 6)
plt.show()

#
#
# plt.scatter(x[:100], y[:100], color = 'red')
# plt.show()
#
# plt.scatter(x[100:200], y[100:200], color = 'blue')
# plt.show()
#
# plt.scatter(x[200:], y[200:], color = 'green')
# plt.show()