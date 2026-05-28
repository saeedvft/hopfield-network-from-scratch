import numpy as np

class HopfieldedNetwork:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        self.weights = np.zeros((self.n, self.n))
        
        for pattern in patterns:
            # Flatten 2D pattern to 1D
            pattern = pattern.flatten().reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
        
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iterations=100, async_update=False):
        """Recall pattern using the update rule: s_i(t+1) = sign(Σ w_ij s_j(t))"""
        # Flatten input pattern to 1D
        pattern = pattern.flatten().copy()
        prev_pattern = pattern.copy()
        
        for iteration in range(max_iterations):
            if async_update:
                for i in range(self.n):
                    local_field = np.dot(self.weights[i], pattern)
                    pattern[i] = 1 if local_field >= 0 else -1
            else:
                local_fields = np.dot(self.weights, pattern)
                pattern = np.where(local_fields > 0, 1, -1) # This is activation
            
            if np.array_equal(pattern, prev_pattern):
                break
            prev_pattern = pattern.copy()
        
        # Reshape back to 2D for display
        pattern_2d = pattern.reshape(5, 5)
        return pattern_2d, iteration

if __name__ == "__main__":
    p1 = np.array([[ 1,  1,  1,  1,  1],
                   [ 1,  1,  1,  1,  1],
                   [-1, -1, -1, -1, -1],
                   [ 1,  1,  1,  1,  1],
                   [ 1,  1,  1,  1,  1]])

    p2 = np.array([[ 1, -1,  1, -1, 1],
                   [ 1,  1,  1,  1, 1],
                   [ 1, -1,  1, -1, 1],
                   [ 1,  1,  1,  1, 1],
                   [ 1, -1,  1, -1, 1]])
    
    p3 = np.array([[ 1, -1, -1, -1,  1],
                   [-1,  1, -1,  1, -1],
                   [-1, -1,  1, -1, -1],
                   [-1,  1, -1,  1, -1],
                   [ 1, -1, -1, -1,  1]])
    
    net = HopfieldedNetwork(25)
    net.train([p1, p2, p3])

    corrupted = np.array([[ 1,  1,  1,  1,  1],
                          [ 1,  1, -1,  1,  1],
                          [-1, -1, -1, -1, -1],
                          [ 1,  1,  1,  1,  1],
                          [ 1,  1,  1,  1,  1]])

    pred, iterations = net.predict(corrupted)
    
    print(f"Recovered pattern:")
    print(pred)
    print(f"\nIterations: {iterations}")