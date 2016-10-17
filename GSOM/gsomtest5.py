import numpy as np
import matplotlib.pyplot as plt
from WGSOM import  GSOM
import pandas as pd
# from sgsom import GSOM as GS

rgen = np.random.RandomState(1234)

X = rgen.random_sample(size=(400, 3))

gsom = GSOM(dims=3, hid = 1,  fd=0.9, sf=0.8, radius=40,max_nodes=1000, min_nodes=100, X = X,nei=True,gaussian=True)
# gsom = GS(dims=3, sf=0.9, fd=0.9, max_nodes=400, min_nodes=5, radius=1.1, nei=False)
for i in range(5):
    print "\n iteration ", (i+1)
    gsom.train_batch(X, 2, 0.5*np.exp(-i/10), i%2)

grid, hits = gsom.predict(X)
x, y = grid.T
plt.scatter(x, y, marker = 's', c= X, edgecolors='none' )
plt.show()

x, y = np.array(gsom.grid.values()).T

hits = np.array(gsom.hits.values())

df = pd.DataFrame(grid, columns=['X', 'Y'])
df.plot(kind='scatter', x='X', y='Y')
plt.show()
#
# plt.scatter(x, y, edgecolors='none', c='white')
# for i, j, t in zip(x,y,hits):
#     plt.text(i, j, t, color = plt.cm.Set1(t*1.0/np.max(hits)), fontsize = 12)
#
# plt.show()
#
