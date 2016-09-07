import numpy as np
import matplotlib.pyplot as plt
from AEGSOM import  GSOM
from sgsom import GSOM as GS

rgen = np.random.RandomState(1234)

X = rgen.random_sample(size=(400, 3))

# gsom = GSOM(dims=3,  hid=3, fd=0.1, sf=0.99, radius=10,max_nodes=1000, min_nodes=0, nei = True)
gsom = GS(dims=3, sf=0.99, fd=0.9, max_nodes=400, min_nodes=5, radius=5)
for i in range(100):
    print "\n iteration ", (i+1)
    gsom.train_batch(X, 2, 0.1*np.exp(-i/10), False)

grid, hits = gsom.predict(X)
x, y = grid.T
plt.scatter(x, y, marker = 's', c= X, edgecolors='none' )
plt.show()

