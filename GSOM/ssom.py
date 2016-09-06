import numpy as np
from growingssom import SOM
import matplotlib.pyplot as plt

X = np.random.random((100, 3))

som = SOM(3, 3)

som.train(X)

print 'trained'

x, y = som.grid.T

plt.scatter(x, y, c = som.neurons, marker = "s", edgecolors="none")
plt.show()

