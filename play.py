import numpy as np
from hopfield import Hopfield

n_dim = 10
n_store = 2
list_of_patterns = [np.sign(np.random.normal(size=n_dim)) for i in range(n_store)]

nn = Hopfield(n_dim=n_dim)
nn.train(list_of_patterns)

print('state', nn.s)
print('overlap', nn.calculate_overlap())
print('state distance', nn.calculate_state_distance())

print('Simulating')

N = 100
for i in range(N):
    nn.update_async()


print('Done')
print('state', nn.s)
print('overlap', nn.calculate_overlap())
print('state distance', nn.calculate_state_distance())



