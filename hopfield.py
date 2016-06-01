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
        self.list_of_patterns = None
        self.w = None
        self.state_distance = np.ones(self.n_dim)

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

        # self.w *= (1.0 / self.n_dim)

        return self.w

    def update(self):
        """
        Updates the network state
        :return: next state
        """

        self.state = np.dot(self.w, self.state)
        return self.state

    def normalize(self):
        """
        Normalizes the state
        :return: the normalize state
        """

        self.state = self.state / np.sum(self.state)

        return self.state

    def calculate_state_distance(self):
        """
        Calcualtes the distance between the state and
        all the patterns

        :return: A state distance vector with the distance between
         the actual state of the system and all the stored patterns
        """

        for index, pattern in enumerate(self.list_of_patterns):
            self.state_distance[index] = np.linalg.norm(self.state - pattern)

        return self.state_distance

