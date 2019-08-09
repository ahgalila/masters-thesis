from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt

import sys
sys.dont_write_bytecode = True

from data import DataLoader
from lstm_model import LSTMModel
from feed_forward_model import FeedForwardModel

k = 5
results  = []
total = 0

for i in range(k):

    dataLoader = DataLoader(src="MNIST_data/dataset")

    [
        noisyTrain,
        noisySymbolsTrain,
        targetsTrain,
        noisyValidation,
        noisySymbolsValidation,
        targetsValidation,
        noisyTest,
        noisySymbolsTest,
        targetsTest
    ] = dataLoader.getParallelDataWithDontCare(i, 2, 2)

    model = FeedForwardModel([512, 512, 512], 3136, 20)
    model.train(noisySymbolsTrain, targetsTrain, noisySymbolsValidation, targetsValidation, 100, 50)
    result = model.evaluate(noisyTest, targetsTest)

    results.append(result[1])
    total += result[1]

print(results)
print("SCORE: " + str(total / k))

#Symbols:
#[0.73, 0.64, 0.73, 0.68, 0.765]
#SCORE: 0.7090000000000001

#Noisy
#[0.585, 0.45, 0.55, 0.55, 0.55]
#SCORE: 0.5369999999999999