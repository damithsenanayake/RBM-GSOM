import numpy as np
from scipy.spatial import distance as dist
class RBM(object):

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def __init__(self, visible, hidden):
        self.vis = visible
        self.hid = hidden
        self.w = np.random.randn(visible, hidden) * 0.1
        self.a = np.zeros((1, visible))
        self.b = np.zeros((1,hidden))
        self.pv = np.zeros((visible,1))
        self.nv = np.zeros((visible,1))
        self.hid_ac = None
        self.ph = np.zeros((hidden,1))
        self.nh = np.zeros((hidden,1))
        self.delta_w = np.zeros(self.w.shape)
        self.delta_a = np.zeros(self.a.shape)
        self.delta_b = np.zeros(self.b.shape)
        self.poshidstates = np.zeros((1,hidden))


    def positive(self, x):
        self.ph = self.sigmoid(x.dot(self.w)+self.b)
        self.posprods = x.T.dot(self.ph)
        self.poshidacts = sum(self.ph)
        self.posvisacts = sum(x)
        self.poshidstates = np.random.binomial(1, self.ph)
        if isinstance(self.poshidstates, int):
            self.poshidstates = np.array([self.poshidstates])

    def reconstruct(self):

        self.negdata = self.sigmoid(self.poshidstates.dot(self.w.T) + self.a)
        self.nh = self.sigmoid(self.negdata.dot(self.w)+self.b)
        self.negprods = self.negdata.T.dot(self.nh)
        self.neghidacts = sum(self.nh)
        self.negvisacts = sum(self.negdata)
        return self.negdata

    def disc_error(self, x):
        ph = self.sigmoid(x.dot(self.w)+self.b)
        hstates = np.random.binomial(1, ph)
        act_hstates = np.random.binomial(1, self.ph)
        return  dist.jaccard(hstates, act_hstates)
        # return np.linalg.norm((hstates - act_hstates))

    def train(self, x, w_cost=0.0001, momentum=0.5 , beta = 0.01, eps =0.00001):#0.008):

        self.positive(x)
        self.reconstruct()

        self.q = self.poshidacts * 1.0 / x.shape[0]
        self.p = np.zeros(self.q.shape)
        self.p.fill(0.0001)
        sparsity_derv = self.q - self.p
        sparsity_matrix = np.repeat(sparsity_derv, self.vis).reshape((self.hid, self.vis)).T

        self.delta_w = momentum * self.delta_w + eps * ((self.posprods - self.negprods) * 1.0/x.shape[0] - w_cost * self.w) + beta * sparsity_matrix
        self.delta_a = momentum * self.delta_a + (eps / x.shape[0]) * (self.posvisacts - self.negvisacts) + beta * sparsity_matrix[:,0]
        self.delta_b = momentum * self.delta_b + (eps / x.shape[0]) * (self.poshidacts - self.neghidacts) + beta * sparsity_derv

        self.w += self.delta_w
        self.a += self.delta_a
        self.b += self.delta_b

