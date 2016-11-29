import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hopfield import Hopfield

n_dim = 5
n_store = 2
p_1 = np.array((1, -1, 1, 1, -1))
p_2 = np.array((1, 1, 1, -1, -1))

list_of_patterns = [p_1, p_2]

nn = Hopfield(n_dim=n_dim)
nn.train(list_of_patterns, normalize=False)

print(nn.w)


