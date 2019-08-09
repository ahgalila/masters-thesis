import numpy as np
from random import randint

testWatchListA = {}
testWatchListB = {}

for a in range(10):
    testWatchListA[str(a)] = randint(0, 9)
for b in range(10):
    testWatchListB[str(b)] = randint(0, 9)

np.save("MNIST_data/unseen_a.npy", testWatchListA)
np.save("MNIST_data/unseen_b.npy", testWatchListB)