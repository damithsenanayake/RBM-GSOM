import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from SelfOrganizingSwarm import SelfOrganizingSwarm

fi = pd.read_csv('../mnist_train.csv', header=None)
test = pd.read_csv('../mnist_test.csv', header=None)
dat =np.array(fi)[:, 1:]/255.0
labels = np.array(fi)[:, 0]
samples = 5000
x, y = SelfOrganizingSwarm(iterations=5).fit_transform(dat[:samples]).T
plt.scatter(x, y, edgecolors='none',c='white')

for i, j, t in zip(x,y, labels[:samples]):
    plt.text(i, j, t, color = plt.cm.Set1(t/10.0), fontsize = 12)

plt.show()

