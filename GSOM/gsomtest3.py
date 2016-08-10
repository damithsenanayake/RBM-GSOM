from sgsom import GSOM
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import KernelPCA,FactorAnalysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
rng = np.random.RandomState(seed=1234)
dat = rng.random_sample((100, 3))
from AutoEncoder import AutoEncoder


fi = pd.read_csv('../mnist_train.csv', header=None)
test = pd.read_csv('../mnist_test.csv', header=None)
dat =np.array(fi)[:10000, 1:]#/255
labels = np.array(fi)[:, 0]
# reductor = AutoEncoder(hid_size=100)
# X = reductor.reduce(dat)

gsom = GSOM(dims=784, sf = 0.3, fd = 0.2, max_nodes = 1500, min_nodes = 100, radius=2)


for i in range(200):
    print '\nbatch '+ str(i+1)
    gsom.train_batch(dat[i*50:(i+1)*50], lr = 0.1, iterations=10)#lr=0.01*np.exp(-i/200), iterations=100)
    # if len(gsom.neurons) > gsom.max_nodes:
    #     gsom.cull_old()

grid, hits = gsom.predict((dat[:500]))
x, y = grid.T
colors = dat
nodes = np.unique(np.array(hits))

batches = []

for n in nodes:
    ninds = np.where(np.array(hits)==n)[0]
    batches.append(dat[ninds])

plt.scatter(x, y, s=1)

for i, j, t in zip(x, y, labels):
    plt.text(i, j, t, color = 'purple', fontsize = 12)
plt.show()
#
# nets = {}
# for n in range(nodes.shape[0]):
#     print 'training : ', nodes[n]
#     nets[nodes[n]] = AutoEncoder(vis=784, hid=100)
#     gsom.range = 5
#     neighborhood = gsom.get_neighbourhood(nodes[n])
#     for n in neighborhood:
#         lr = gsom.grid[n] - gsom.
#     nets[nodes[n]].train(batches[n]/255.0, 10000, 0.1)#,batch_size=
#                  #2)
#
# predictions = []
#
# for net in nets.keys():
#     pred_n = nets[net].predict(dat[:1000]/255.0)
#     err = np.linalg.norm(pred_n - dat[:1000]/255.0, axis=1)
#     predictions.append(err)
#
# errors = np.array(predictions)
#
# x =[]
# y = []
#
# for i in range(errors.shape[1]):
#     node = gsom.grid[nodes[np.argmin(errors[:, i])]]
#     x.append(node[0])
#     y.append(node[1])
#
#
# plt.scatter(x, y, s=1)
#
# for i, j, t in zip(x, y, labels):
#     plt.text(i, j, t, color = 'purple', fontsize = 12)
# plt.show()