import numpy as np

class HopfieldedNetwork(self, n_neurons):
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
    


if __name__ == "__main__":
    p1 = np.array([1, 1, -1, -1, 1, -1])
    p2 = np.array([-1, 1, 1, -1, 1, -1])

    hopfield_network = HopfieldedNetwork(6)
    hopfield_network.train([p1, p2])