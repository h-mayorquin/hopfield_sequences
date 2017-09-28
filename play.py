import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hopfield import Hopfield, HopfieldSequence, HopfieldDiff

prng = np.random.RandomState(seed=101)

n_dim = 10
n_store = 4
T = 0.0

dt = 0.1
tau_m = 20


nn = HopfieldDiff(n_dim=n_dim, tau_m=tau_m, dt=dt, T=T, prng=prng)
list_of_patterns = nn.generate_random_patterns(n_store)
nn.train(list_of_patterns, normalize=True)

N = 1000
history = np.zeros((N, n_dim))
distance_history = np.zeros((N, 2, n_store))
overlap_history = np.zeros((N, n_store))

for i in range(N):
    nn.update()
    history[i, ...] = nn.s
    distance_history[i, ...] = nn.calculate_state_distance()
    overlap_history[i, ...] = nn.calculate_overlap()


# Let's plot the distance history
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)

for i in range(n_store):
    ax.plot(distance_history[:, 0, i])
    ax.plot(distance_history[:, 1, i])


ax.axhline(0, color='black')

plt.show()
