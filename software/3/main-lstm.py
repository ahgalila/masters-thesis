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
    ] = dataLoader.getSequencialDataWithDontCare(i, 2, 2)

    model = LSTMModel(2, [512, 512, 512], 1568, 20)
    model.train(noisyTrain, targetsTrain, noisyValidation, targetsValidation, 100, 50)
    result = model.evaluate(noisyTest, targetsTest)

    results.append(result[1])
    total += result[1]

print(results)
print("SCORE: " + str(total / k))