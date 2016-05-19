import numpy as np
from hopfield import Hopfield


s0 = np.array([1, 0, 0])
s1 = np.array([0, 1, 0])

s = np.array([0.5, 0.3, 8])

network = Hopfield(s)

list_of_patterns = [s0, s1]
w = network.train(list_of_patterns)

N = 3
for i in range(N):
    print(network.update())