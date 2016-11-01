
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
import sys

class SelfOrganizingSwarm(object):

    def __init__(self, dim_out=2, verbose=0,iterations=1000):
        self.d = dim_out
        self.verbose = verbose
        self.grid = []
        self.iterations = iterations
        self.G = None
        self.P = None
        self.ts = 0
        self.fs = 0.00000001
        self.neighbors = None

    def get_neighbors(self, point):
        if not np.random.binomial(1,self.ts*1.0 / (self.ts + self.fs) ) or self.G == None:#self.ts*1.0 / (self.ts + self.fs)<0.75:
            self.G = self.triangulate()
        if np.all(self.G==self.P):
            self.ts +=1
        else :
            self.fs += 1
            self.P = self.G
        return np.where(self.G[point])[0]
        # return np.linalg.norm(self.grid - self.grid[point], axis=1).argsort()[:5]

    def triangulate(self):
        adj = np.zeros((self.grid.shape[0], self.grid.shape[0]))
        tri = Delaunay(self.grid)
        chull = ConvexHull(self.grid).simplices
        #     print chull
        for simplegrid in tri.simplices:
            for vert in simplegrid:
                for vert2 in simplegrid:
                    adj[vert][vert2] = 1
        # for line in chull:
        #     adj[line[0]][line[1]]=0
        #     adj[line[1]][line[0]]=0

        return adj

    def fit(self, X):
        for i in range(int(np.ceil(np.sqrt(X.shape[0])))):
            for j in range(int(np.ceil(np.sqrt(X.shape[0])))):
                self.grid.append([i, j])

        self.grid = np.array(self.grid).astype(float)

        self.grid = np.random.random((X.shape[0]  , self.d))
        self.C = np.ones((self.grid.shape[0], X.shape[1])) * X.mean()
        verbosity_limit = self.iterations * self.verbose
        for it in range(self.iterations):
            # if not (it % verbosity_limit):
            sys.stdout.write('\r iteration %s' % str(it + 1))
            for x in X[np.random.permutation(X.shape[0])]:
                # find BMU
                bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
                neighbors = self.get_neighbors(bmu)
                self.neighbors = neighbors
                dists = np.linalg.norm(self.grid[neighbors] - self.grid[bmu], axis=1)
                rad = dists.max()
                moving_amounts =  np.array([np.exp(-0.5*dists**2/rad**2)]).T#/np.sum(np.exp(-dists**2/rad**2)) #/((np.sum(np.array([np.exp(-dists)]).T) ) * dists.shape[0])

                self.C[neighbors] += 0.5 * (x - self.C[neighbors]) * np.exp(-it / (self.iterations))  * moving_amounts


            for x in X:
                bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
                neighbors = self.get_neighbors(bmu)
                self.neighbors = neighbors
                dists = np.linalg.norm(self.C[neighbors] - x, axis=1)

                mean = dists.mean()
                variance = dists.var()
                rad = dists.max()
                inds = np.where((dists - mean)/np.sqrt(variance) >2.05)[0]
                far_ones = neighbors[inds]
                close_ones = np.setdiff1d(neighbors, far_ones)
                nei_inds = np.setdiff1d(np.array(range(dists.shape[0])), inds)
                try:
                    nei_dists =dists[nei_inds]
                except:
                    continue
                neighbors = close_ones
                not_neighbors = far_ones
                moving_amounts = np.array([np.exp(-0.5*nei_dists**2/rad**2)]).T #/ nei_dists.shape[0]

                moving_directions = self.grid[bmu] - self.grid[neighbors]
                self.grid[neighbors] += 0.25*np.exp(-it / (1 * self.iterations)) * moving_amounts * moving_directions

                not_nei_dists = dists[inds]
                if not not_nei_dists.shape[0] == 0:
                    not_nei_dists/=rad
                moving_amounts =np.array([np.exp(-0.5*not_nei_dists**2)]).T#/ not_nei_dists.shape[0]

                moving_directions = self.grid[bmu] - self.grid[not_neighbors]
                self.grid[not_neighbors] -= 0.25*np.exp(-it / (1 * self.iterations)) * moving_amounts * moving_directions

    def predict(self, X):
        out = []
        for x in X:
            bmu = np.argmin(np.linalg.norm(self.C - x, axis=1))
            out.append(self.grid[bmu])
        out = np.array(out)
        return out


    def fit_transform(self, X):
        self.fit(X)
        return self.predict(X)