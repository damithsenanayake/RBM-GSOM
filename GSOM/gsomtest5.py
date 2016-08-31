import numpy as np
import matplotlib.pyplot as plt
from AEGSOM import GSOM

rgen = np.random.RandomState(1234)

X = rgen.random_sample(size=(400, 3))

gsom = GSOM(dims=3,  hid=4, fd=0.1, sf=0.1, radius=1.1,max_nodes=1000, min_nodes=300)

for i in range(10):
    print "\n iteration ", (i+1)
    gsom.train_batch(X, 10, 0.1/(i+1), True)

grid, hits = gsom.predict(X)
x, y = grid.T
plt.scatter(x, y, marker = 7, c= X )
plt.show()

