import numpy as np
import sys
class SOM(object):
    def __init__(self, dims, grid_side):
        self.neurons = np.random.random((grid_side * grid_side, dims))
        grid = []
        # for x in range(grid_side):
        #     for y in range(grid_side):
        #         grid.append([x, y])
        # self.grid = np.array(grid).astype(float)
        self.errors = np.zeros(grid_side*grid_side)
        self.GT = - dims * np.log(0.001)
        self.gamma = 4
        self.grid = np.random.random((grid_side * grid_side, 2))


    def find_bmu(self, x):

        diffs = (self.neurons - x)
        dists = np.linalg.norm(diffs, axis=1)

        bmu = np.argmin(dists)

        return bmu, dists.min()


    def get_neighborhood(self, node, num):

        coords = self.grid[node]

        diffs = self.grid - coords
        dists = np.linalg.norm(diffs, axis=1)
        nodes = np.where(dists < num)[0]
        return nodes, dists[nodes]
        #return np.argsort(dists)[:int(num)], np.sort(dists)[:int(num)]

    def predict(self, X):
        out = []
        for x in X:
            out.append(self.find_bmu(x)[0])
        return np.array(out)

    def train(self, X):
        rad = 0.4

        for b in range(0,100):
            for x in X:
                sys.stdout.write("\r iter %i nodes %i"%(b+1, self.neurons.shape[0]))
                sys.stdout.flush()
                bmu, error = self.find_bmu(x)
                neighbors, dists = self.get_neighborhood(bmu, rad)
                if dists.shape[0] == 0:
                    return
                m = max(dists)
                h = np.exp(-dists ** 2 / m ** 2)
                self.errors[bmu] += error
                self.neurons[neighbors] += 0.01*(x - self.neurons[neighbors] ) * np.array([h]).T

                neighbors, dists = self.get_neighborhood(bmu, 70)
                m = max(dists)
                h = np.exp(-dists ** 2 / m ** 2)

                self.grid[neighbors] += 0.13*(self.grid[bmu] - self.grid[neighbors])*np.array([h]).T/h.sum()

                if self.errors[bmu] > self.GT:
                    for i in range(int(neighbors.shape[0] - self.gamma)):
                        new_node = rad * np.random.randn(1, 2) + self.grid[bmu]
                        new_weight = np.mean(self.neurons[self.get_neighborhood(bmu,rad)[0]], axis = 0)
                        self.neurons = np.append(self.neurons, np.array([new_weight]), axis = 0)
                        self.grid = np.append(self.grid, new_node, axis=0)
                        self.errors = np.append(self.errors, 0)

                    neighbors, dists = self.get_neighborhood(bmu, rad)
                    m = max(dists)
                    h = np.exp(-dists ** 2 / m ** 2)
                    self.errors[neighbors] += self.errors[bmu] * 0.001 * h / h.sum()

            rad *= 0.99
            self.gamma -=1



