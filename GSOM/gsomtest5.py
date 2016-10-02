import numpy as np
import matplotlib.pyplot as plt
from sgsom import  GSOM
# from sgsom import GSOM as GS

rgen = np.random.RandomState(1234)

X = rgen.random_sample(size=(400, 3))

gsom = GSOM(dims=3,  fd=0.9, sf=0.8, radius=5,max_nodes=1000, min_nodes=0)
# gsom = GS(dims=3, sf=0.9, fd=0.9, max_nodes=400, min_nodes=5, radius=1.1, nei=False)
for i in range(10):
    print "\n iteration ", (i+1)
    gsom.train_batch(X, 2, 0.5*np.exp(-i/10), False)

grid, hits = gsom.predict(X)
x, y = grid.T
plt.scatter(x, y, marker = 's', c= X, edgecolors='none' )
plt.show()

