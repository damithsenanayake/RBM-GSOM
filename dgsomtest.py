from dulini_gsom import GSOM
import numpy as np

X = np.random.random((100,3))

gsom = GSOM(3, 0.5, 0.7, 0.7, 5)

for x in X:
    gsom.train(x)
