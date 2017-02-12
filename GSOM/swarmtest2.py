import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MovingMap import  MovingMap
from bgsom import GSOM
from sklearn.manifold import MDS
from SelfOrganizingSwarm import SelfOrganizingSwarm

fi = pd.read_csv('../mnist_train.csv', header=None)
test = pd.read_csv('../mnist_test.csv', header=None)
dat =np.array(fi)[:, 1:]#/255.0
labels = np.array(fi)[:, 0]
samples = 500
# x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T

# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
x, y = GSOM().fit_transform(dat[:samples], beta=0.75, sf=0.05, lr = 0.5).T

# x, y = MDS().fit_transform(dat[:samples]).T
fig = plt.figure()

plt.subplot(211)

plt.scatter(x, y, edgecolors='none',c='white')

for i, j, t in zip(x,y, labels[:samples]):
    plt.text(i, j, t, color = plt.cm.Set1(t/10.0), fontsize = 5, alpha = 0.75)
plt.subplot(212)

plt.scatter(x, y , edgecolors='none', c = 'grey', alpha = 0.8, s = 15)

plt.show()

