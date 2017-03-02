import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MovingMap import  MovingMap
from bgsom import GSOM
from sklearn.preprocessing import normalize
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from SelfOrganizingSwarm import SelfOrganizingSwarm


fi = pd.read_csv('../mnist_train.csv', header=None)
test = pd.read_csv('../mnist_test.csv', header=None)
dat =np.array(fi)[:, 1:]/255.0
labels = np.array(fi)[:, 0]
samples = 6000
# x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T

# x, y = MovingMap(iterations=100, beta=1.5).fit_transform(dat[:samples]).T
Y= GSOM().fit_transform(dat[:samples], lr = 1.0, beta=0.2, sf=0.8, fd = 0.9, wd=0.0225)
# Y = TSNE().fit_transform(dat[:samples])
x, y = Y.T
# x, y = MDS().fit_transform(dat[:samples]).T
fig = plt.figure()

kl = KMeans(10).fit(Y).labels_

print 'ars :', adjusted_rand_score(labels[:samples], kl)
print 'ami :', adjusted_mutual_info_score(labels[:samples], kl)
plt.subplot(211)

plt.scatter(x, y, edgecolors='none',c=plt.cm.jet(labels/10.0), alpha = 0.5, s = 15)
plt.subplot(212)

plt.scatter(x, y , edgecolors='none', c = 'grey', alpha = 0.5, s = 15)

plt.show()

