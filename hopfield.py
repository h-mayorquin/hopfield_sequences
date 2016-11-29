import numpy as np


class Hopfield():
    """
    The hopfield network in the simplest case of AMIT book in attractor neural network
    """

    def __init__(self, n_dim=3):

        self.n_dim = n_dim
        self.s = np.sign(np.random.normal(size=n_dim))
        self.h = np.zeros(n_dim)

        self.list_of_patterns = None
        self.w = None
        self.m = None
        self.state_distance = None

    def train(self, list_of_patterns):
        """
        Implements the Hebbian learning rule

        :param list_of_patterns: This is a list with the desired parameters for equilibrum
        :return: w  the weight matrix.
        """

        self.list_of_patterns = list_of_patterns
        self.w = np.zeros((self.n_dim, self.n_dim))

        for pattern in list_of_patterns:
            self.w += np.outer(pattern, pattern)

        self.w *= (1.0 / self.n_dim)

    def update_sync(self):
        """
        Updates the network state of all the neurons at the same time
        """
        # Linear part
        self.h = np.dot(self.w, self.s)
        # Non-linear part
        # self.state = sigmoid_logistic(self.state)
        self.s = np.sign(self.h)

    def update_async(self):
        """
        Updates the network state one neuron at a time
        """
        # Generate random number
        i = np.random.randint(self.n_dim, size=1)
        # Linear
        # self.state = np.dot(self.state, self.w[i, ...])
        self.h[i] = np.dot(self.w[i, ...], self.s)
        # Non-linear
        self.s[i] = np.sign(self.s[i])

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

