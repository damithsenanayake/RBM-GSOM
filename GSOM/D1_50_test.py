import pandas as pd
import numpy as np
from AEGSOM import GSOM
import matplotlib.pyplot as plt

data = pd.read_csv('~/data/D1_50.csv', header=None)

X= np.array(data)[:, :-4]
print X.shape
Y = np.array(data)[:, -1]


gsom = GSOM(dims=50,  hid=2, fd=0.1, sf=0.1, radius=15,max_nodes=2005, min_nodes=0)

for i in range(100):
    print "\n iteration ", (i+1)
    gsom.train_batch(X, 1, 0.01*np.exp(-i/10), True)

grid, hits = gsom.predict(X)
x, y = grid.T
plt.scatter(x, y, marker = 's', edgecolors='none', c = Y )
plt.show()
