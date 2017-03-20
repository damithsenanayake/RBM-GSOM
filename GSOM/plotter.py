import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tsne = np.array(pd.read_csv('tsne-6000-mnist.csv'))
tem = np.array(pd.read_csv('tem-6000-mnist.csv'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Cluster Quality Metric')
ax.set_ylabel('Score')
bp1 = ax.boxplot(tem, labels=np.array(['ARS', 'AMI']),showfliers=False)
ax.boxplot(tsne, labels=np.array(['ARS', 'AMI']),  vert=1, whis=1.5, notch=0, sym='+',showfliers=False)

plt.setp(bp1['boxes'], color = 'blue')
plt.setp(bp1['whiskers'], color = 'blue')
plt.setp(bp1['caps'], color = 'blue')
plt.setp(bp1['medians'], color = 'green')
# plt.setp(bp1['fliers'], color = 'red', marker = '+')

plt.show()