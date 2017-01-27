import numpy as np
import matplotlib.pyplot as plt

a = 10
b = 10
y=[]
x=[]
z = []
it = 40
for i in range(it):

    y.append(a *np.exp(-7.5*i/it))
    # b *= np.exp(-0.1*i**2/it**2)
    # z.append(b)
    x.append(i)

plt.subplot(211)
plt.plot(x, y, c = 'blue')
# plt.subplot(212)
# plt.plot(x, z, c = 'red')
plt.show()