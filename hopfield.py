import numpy as np


class Hopfield():
    """
    The hopfield network in the simplest case of AMIT book in attractor neural network
    """

    def __init__(self, n_dim=3, T=0, prng=np.random):

        self.prng = prng

        self.n_dim = n_dim
        self.s = np.sign(prng.normal(size=n_dim))
        self.h = np.zeros(n_dim)

        # Noise parameters
        self.T = T
        self.sigma = T / (2 * np.sqrt(2))  # Check page 67 of Amit to see where does this comes from

        self.list_of_patterns = None
        self.w = None
        self.m = None
        self.state_distance = None

    def train(self, list_of_patterns, normalize=True):
        """
        Implements the Hebbian learning rule

        :param list_of_patterns: This is a list with the desired parameters for equilibrum
        normalize: normalizes the w matrix by its dimension
        :return: w  the weight matrix.
        """

        self.list_of_patterns = list_of_patterns
        self.w = np.zeros((self.n_dim, self.n_dim))

        for pattern in list_of_patterns:
            self.w += np.outer(pattern, pattern)

        if normalize:
            self.w *= (1.0 / self.n_dim)

        # zeros in the diagonal
        self.w[np.diag_indices_from(self.w)] = 0

    def generate_random_patterns(self, n_store):

        list_of_patterns = [np.sign(self.prng.normal(size=self.n_dim)) for i in range(n_store)]

        return list_of_patterns

    def update_sync(self):
        """
        Updates the network state of all the neurons at the same time
        """

        if self.sigma < 0.001:
            noise = 0
        else:
            noise = self.prng.normal(0, scale=self.sigma, size=self.n_dim)

        # Linear part
        self.h = np.dot(self.w, self.s) + noise
        # Non-linear part
        # self.state = sigmoid_logistic(self.state)
        self.s = np.sign(self.h)

    def update_async(self):
        """
        Updates the network state one neuron at a time
        """
        # Generate random number
        i = self.prng.randint(self.n_dim, size=1)[0]
        # Linear
        # self.state = np.dot(self.state, self.w[i, ...])

        if self.sigma < 0.001:
            noise = 0
        else:
            noise = self.prng.normal(loc=self.sigma)

        self.h[i] = np.dot(self.w[i, ...], self.s) + noise
        # Non-linear
        self.s[i] = np.sign(self.h[i])

    def calculate_overlap(self):
        self.m = np.mean(self.s * self.list_of_patterns, axis=1)

        return self.m

    def calculate_state_distance(self):
        """
        Calcualtes the distance between the state and
        all the patterns

        :return: A state distance vector with the distance between
         the actual state of the system and all the stored patterns
        """

        self.state_distance = np.ones(len(self.list_of_patterns))

        for index, pattern in enumerate(self.list_of_patterns):
            self.state_distance[index] = np.linalg.norm(self.s - pattern)

        return self.state_distance

