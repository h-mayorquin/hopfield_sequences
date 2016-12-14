import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hopfield import Hopfield, HopfieldSequence, HopfieldDiff

prng = np.random.RandomState(seed=101)

n_dim = 10
n_store = 3
T = 0.0

dt = 0.1
tau_m = 20


nn = HopfieldDiff(n_dim=n_dim, tau_m=tau_m, dt=dt, T=T, prng=prng)
list_of_patterns = nn.generate_random_patterns(n_store)
nn.train(list_of_patterns, normalize=True)

N = 1000
history = np.zeros((N, n_dim))

for i in range(N):
    nn.update()
    history[i, ...] = nn.s

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)

for s in history.T:
    ax.plot(s)

plt.show()
