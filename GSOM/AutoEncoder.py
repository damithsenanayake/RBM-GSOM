import numpy as np
import time
import sys

class AutoEncoder(object):

    def __init__(self, vis, hid):
        self.w1 = np.random.random((784,100)) * 2 -1
        self.w2 = self.w1.T
        self.a = np.zeros(hid)
        self.b = np.zeros(vis)
        self.st = time.time()
        self.mw1 = np.zeros(self.w1.shape)
        self.mw2 = np.zeros(self.w2.shape)
        self.ma = np.zeros(self.a.shape)
        self.mb = np.zeros(self.b.shape)

    def sig(self, X):
        return 1/(1+np.exp(-X))

    def train_batch(self,X, iters, eps, momentum = 0.75, batch_size=10):
        ret = None

        if X.shape[0] > batch_size:
            for i in range(X.shape[0]/batch_size):
                ret =self.train(X[i*batch_size:(i+1)*batch_size], iters, eps * np.exp(-i*1.0/(X.shape[0]/batch_size)), momentum)
                sys.stdout.write('\rbatch %i of %i'%(i+1, X.shape[0]/batch_size))
                sys.stdout.flush()
                momentum *= 0.8
        print ""
        return ret
    def train(self, X, iters, eps, momentum=0.5, wd_param = 0.1):
        for i in range(iters):
            l1 = self.sig(np.dot(X, self.w1) + self.a)  # 1/(1+np.exp(-(np.dot(X,syn0))))
            l2 = self.sig(np.dot(l1, self.w2) + self.b)  # 1/(1+np.exp(-(np.dot(l1,syn1))))
            l2_delta = (X - l2) * (l2 * (1 - l2))
            l1_delta = l2_delta.dot(self.w2.T) * (l1 * (1 - l1))

            #sparsity :
            avg = np.average(l1, axis=0)
            targ = np.ones(avg.shape) * 0.01
            sparse_term = (- targ/avg + (1-targ)/(1-avg)) * (l1 * (1-l1))
            l1_delta += 0.01*sparse_term

            dw2 = eps * (l1.T.dot(l2_delta) + momentum *self.mw2 - wd_param*self.w2)
            dw1= eps * (X.T.dot(l1_delta) + momentum*self.mw1 - wd_param*self.w1)
            da= eps * (l1_delta.sum(axis=0) + momentum*self.ma- wd_param * self.ma)
            db= eps * (l2_delta.sum(axis=0) + momentum*self.mb - wd_param * self.mb)
            self.w2 +=dw2
            self.w1 +=dw1
            self.a +=da
            self.b +=db
            self.mw1 = dw1
            self.mw2 = dw2
            self.ma = da
            self.mb = db
        return l2

    def predict(self, X):
        l1 = self.sig(np.dot(X, self.w1) + self.a)  # 1/(1+np.exp(-(np.dot(X,syn0))))
        l2 = self.sig(np.dot(l1, self.w2) + self.b)
        return l2

