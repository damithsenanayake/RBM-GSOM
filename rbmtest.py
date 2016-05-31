import numpy as np
from RBM import RBM
from matplotlib import pyplot as plt
data = np.loadtxt('mnist_train.csv', delimiter=',')

def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """

    figure, axes = plt.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    index = 0

    for axis in axes.flat:

        """ Add row of weights as an image to the plot """

        image = axis.imshow(opt_W1[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap = plt.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """

    plt.show()

train = data[:, 1:]/255
train[np.where(train>0)]=1
#works well for binary images, but features don't get properly extraced for non binary images.
ones = train[np.where(np.where(data[:,0] == 5)[0]<100000)[0]]
rbm = RBM(784, 100)
m = 0.5
for i in range(10):
    if i > 5:
        m = 0.9
    n = 10
    for j in range(1000):
        rbm.train(train[j*10:j*10+9],  momentum=m, w_cost=0.0001)

w = rbm.w#.flatten()

visualizeW1(w.T, 28, 10)
#
# plt.imshow(np.reshape(ones[20], (-1,28)))
# plt.show()
#
rbm.train(ones[:10], momentum= m, w_cost=0.0001)
plt.imshow(np.reshape(ones[7], (-1,28)))
plt.show()
plt.imshow(np.reshape(rbm.negdata[7],(-1, 28)))
plt.show()