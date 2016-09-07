import numpy as np
import pandas as pd
from AEGSOM import  GSOM
import matplotlib.pyplot as plt
data = pd.read_csv("~/data/5D3C_RAW.csv", header=None)

X = np.array(data)
# add noise

noise = np.random.randn(300, 10)

# X = np.append(X,noise, axis = 1)

indices = np.random.permutation(X.shape[0])
gsom = GSOM(dims=10, hid = 5, sf = 0.00001, fd = 0.00001, max_nodes = 1000, min_nodes = 0, radius=10, X = X, gaussian=False)

for i in range(50):
    print "\nbatch ", (i+1)
    gsom.train_batch(X[indices], iterations=2, lr = 1*np.exp(-i/50), prune=True)

# gsom.prune()


grid, hits = gsom.predict(X)
x, y = grid.T

plt.scatter(X.T[0], X.T[1])
plt.show()

plt.scatter(x[:100], y[:100], color = 'red',marker = 4)
plt.scatter(x[100:200], y[100:200], color = 'blue', marker = 5)
plt.scatter(x[200:], y[200:], color = 'green', marker = 6)
plt.show()

plt.scatter(x[:100], y[:100], color = 'red')
plt.show()

plt.scatter(x[100:200], y[100:200], color = 'blue')
plt.show()

plt.scatter(x[200:], y[200:], color = 'green')
plt.show()