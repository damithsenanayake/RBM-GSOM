import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from SelfOrganizingSwarm import SelfOrganizingSwarm

fi = pd.read_csv('../mnist_train.csv', header=None)
test = pd.read_csv('../mnist_test.csv', header=None)
dat =np.array(fi)[:, 1:]/255.0
labels = np.array(fi)[:, 0]
samples = 5000
x, y = SelfOrganizingSwarm(iterations=10, alpha=1, beta=0.1, delta=0, theta=3.5).fit_transform(dat[:samples]).T
plt.scatter(x, y, edgecolors='none',c='white')

for i, j, t in zip(x,y, labels[:samples]):
    plt.text(i, j, t, color = plt.cm.Set1(t/10.0), fontsize = 10, alpha = 0.5)

plt.show()

