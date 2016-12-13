import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hopfield import Hopfield, HopfieldSequence

prng = np.random.RandomState(seed=100)

n_dim = 10
n_store = 5
T = 0.0
tau = 10


if True:
    list_of_patterns = [np.sign(np.random.normal(size=n_dim)) for i in range(n_store)]
else:
    list_of_patterns = [-1 * np.ones(n_dim) for i in range(n_store)]
    for i in range(n_store):
        list_of_patterns[i][i] = 1

list_of_patterns_sequence = list_of_patterns[:3]

nn = HopfieldSequence(n_dim=n_dim, tau=tau, g_delay=2.0, T=T, prng=prng)
nn.train(list_of_patterns, normalize=True)
nn.train_delays(list_of_patterns_sequence, normalize=True)

N = 20

nn.s = np.copy(list_of_patterns[0])
history = np.zeros((N, n_store))
for i in range(N):
    nn.update_async()
    history[i, :] = nn.calculate_overlap()


# Plot
if True:
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)

    for index, overlap in enumerate(history.T):
        ax.plot(overlap, '-*', label=str(index))

    ax.set_ylim([-1.2, 1.2])
    ax.legend()
    plt.show()