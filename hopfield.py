import numpy as np


class Hopfield():
    """
    The basic construct of the Hopfield network.
    """

    def __init__(self, s):
        """
        This is the basis constructor
        """

        self.state = s
        self.n_dim = s.shape[0]

    def train(self, list_of_patterns):
        """
        Implements the Hebbian learning rule

        :param list_of_patterns: This is a list with the desired parameters for equilibrum
        :return: w  the weight matrix.
        """

        self.w = np.zeros((self.n_dim, self.n_dim))
        for pattern in list_of_patterns:
            self.w += np.outer(pattern, pattern)

        self.w *= (1.0 / self.n_dim)

        return self.w

    def update(self):
        """
        Updates the network state
        :return: next state
        """

        self.state = np.dot(self.w, self.state) > 0
        return self.state * 1.0

