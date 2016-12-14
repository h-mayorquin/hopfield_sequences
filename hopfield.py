import numpy as np
import collections


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


class HopfieldSequence():
    """
    The hopfield as a sequence
    """

    def __init__(self, n_dim=3, tau=10, g_delay=1.0, T=0, prng=np.random):
        self.prng = prng

        self.n_dim = n_dim
        self.tau = tau
        self.g_delay = g_delay
        self.s = np.sign(prng.normal(size=n_dim))
        self.h = np.zeros(n_dim)

        # Noise parameters
        self.T = T
        self.sigma = T / (2 * np.sqrt(2))  # Check page 67 of Amit to see where does this comes from

        self.list_of_patterns = None
        self.w = None
        self.w_delay = None
        self.m = None
        self.state_distance = None

        aux = [np.zeros(n_dim) for i in range(self.tau)]
        self.s_history = collections.deque(aux, maxlen=self.tau)

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

    def train_delays(self, list_of_patterns, normalize=True):

        self.list_of_patterns_sequence = list_of_patterns
        self.w_delay = np.zeros((self.n_dim, self.n_dim))

        for index in range(len(list_of_patterns) - 1):
            pattern1 = list_of_patterns[index + 1]
            pattern2 = list_of_patterns[index]
            self.w_delay += np.outer(pattern1, pattern2)

        if normalize:
            self.w_delay *= (1.0 / self.n_dim)

        # zeros in the diagonal
        self.w_delay[np.diag_indices_from(self.w_delay)] = 0

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
        self.h = np.dot(self.w, self.s) + \
                 self.g_delay * np.dot(self.w_delay, self.s_history.pop()) + noise

        # Non-linear part
        # self.state = sigmoid_logistic(self.state)
        self.s = np.sign(self.h)
        self.s_history.appendleft(np.copy(self.s))

    def update_async_random_sequence(self):
        random_sequence = self.prng.choice(self.n_dim, size=self.n_dim, replace=False)
        for i in random_sequence:
            self.update_async_one(i)

        self.s_history.appendleft(np.copy(self.s))

    def update_async_one(self, i=None):

        if i is None:
            i = self.prng.randint(self.n_dim, size=1)[0]

            # Linear
            # self.state = np.dot(self.state, self.w[i, ...])

        if self.sigma < 0.001:
            noise = 0
        else:
            noise = self.prng.normal(loc=self.sigma)

        self.h[i] = np.dot(self.w[i, ...], self.s) \
                    + self.g_delay * np.dot(self.w_delay[i, ...], self.s_history[-1]) + noise
        # Non-linear
        self.s[i] = np.sign(self.h[i])

    def update_async(self, i=None):
        """
        Updates the network state one neuron at a time
        """
        # Generate random number
        if i is None:
            i = self.prng.randint(self.n_dim, size=1)[0]

            # Linear
            # self.state = np.dot(self.state, self.w[i, ...])

        if self.sigma < 0.001:
            noise = 0
        else:
            noise = self.prng.normal(loc=self.sigma)

        self.h[i] = np.dot(self.w[i, ...], self.s) \
                    + self.g_delay * np.dot(self.w_delay[i, ...], self.s_history[-1]) + noise
        # Non-linear
        self.s[i] = np.sign(self.h[i])
        self.s_history.appendleft(np.copy(self.s))

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


class HopfieldDiff():

    def __init__(self, n_dim=3, tau_m=20.0, dt=0.1, T=0, prng=np.random):

        self.prng = prng

        self.tau_m = tau_m
        self.dt = dt

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

    def update(self):
        """
        Updates the network state of all the neurons at the same time
        """

        if self.sigma < 0.001:
            noise = 0
        else:
            noise = self.prng.normal(0, scale=self.sigma, size=self.n_dim)

        aux = np.dot(self.w, np.tanh(self.s))
        aux = np.ones(self.n_dim) * 2
        self.s += (self.dt / self.tau_m) * (aux - self.s + noise)


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
