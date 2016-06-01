import numpy as np
from hopfield import Hopfield


s0 = np.array([1, 0])
s1 = np.array([0, 1])

s = np.array([0.9, 0.1])

network = Hopfield(s)

list_of_patterns = [s0, s1]
w = network.train(list_of_patterns)

print(w)
print(network.w)
print('Running')

N = 5
for i in range(N):
    print('state', network.state)
    print('update', network.update())
    print('normalize', network.normalize())
    print('state distance', network.calculate_state_distance())

