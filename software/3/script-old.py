from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt

import sys
sys.dont_write_bytecode = True

from data import DataLoader
from lstm_model import LSTMModel
from feed_forward_model import FeedForwardModel

dataLoader = DataLoader(src="MNIST_data/dataset")

k = 1

'''[
    noisyTrain,
    noisySymbolsTrain,
    targetsTrain,
    noisyValidation,
    noisySymbolsValidation,
    targetsValidation,
    noisyTest,
    noisySymbolsTest,
    targetsTest
] = dataLoader.getSequencialData(4, 2, 2)

resultsNoisy = []
minNoisy = 1.0
maxNoisy = 0.0
sumNoisy = 0.0

for i in range(k):
    

    model = LSTMModel(2, [512, 512], 784, 20)
    model.train(noisyTrain, targetsTrain, noisyValidation, targetsValidation, 100, 200)
    result = model.evaluate(noisyTest, targetsTest)
    sumNoisy += result[1]
    if result[1] < minNoisy:
        minNoisy = result[1]
    if result[1] > maxNoisy:
        maxNoisy = result[1]
    resultsNoisy.append(result[1])

print("Sequential - Noisy")
print("RESULTS: " + str(resultsNoisy))
print("SCORE: " + str(sumNoisy / k))
print("MIN: " + str(minNoisy))
print("MAX: " + str(maxNoisy))'''

'''resultsNoisySymbols = []
minNoisySymbols = 1.0
maxNoisySymbols = 0.0
sumNoisySymbols = 0.0

for i in range(k):
    

    model = LSTMModel(2, [512, 512], 794, 20)
    model.train(noisySymbolsTrain, targetsTrain, noisySymbolsValidation, targetsValidation, 100, 200)
    result = model.evaluate(noisySymbolsTest, targetsTest)
    sumNoisySymbols += result[1]
    if result[1] < minNoisySymbols:
        minNoisySymbols = result[1]
    if result[1] > maxNoisySymbols:
        maxNoisySymbols = result[1]
    resultsNoisySymbols.append(result[1])

print("Sequential - Noisy and Symbols")
print("RESULTS: " + str(resultsNoisySymbols))
print("SCORE: " + str(sumNoisySymbols / k))
print("MIN: " + str(minNoisySymbols))
print("MAX: " + str(maxNoisySymbols))'''

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
] = dataLoader.getSequencialDataWithDontCare(4, 2, 2, 300)

'''resultsNoisyDontCare = []
minNoisyDontCare = 1.0
maxNoisyDontCare = 0.0
sumNoisyDontCare = 0.0

for i in range(k):
    

    model = LSTMModel(2, [512, 512], 794, 20)
    model.train(noisyTrain, targetsTrain, noisyValidation, targetsValidation, 100, 200)
    result = model.evaluate(noisyTest, targetsTest)
    sumNoisyDontCare += result[1]
    if result[1] < minNoisyDontCare:
        minNoisyDontCare = result[1]
    if result[1] > maxNoisyDontCare:
        maxNoisyDontCare = result[1]
    resultsNoisyDontCare.append(result[1])'''

resultsNoisySymbolsDontCare = []
minNoisySymbolsDontCare = 1.0
maxNoisySymbolsDontCare = 0.0
sumNoisySymbolsDontCare = 0.0

for i in range(k):
    

    model = LSTMModel(2, [512, 512], 794, 20)
    model.train(noisySymbolsTrain, targetsTrain, noisySymbolsValidation, targetsValidation, 100, 50)
    result = model.evaluate(noisySymbolsTest, targetsTest)
    sumNoisySymbolsDontCare += result[1]
    if result[1] < minNoisySymbolsDontCare:
        minNoisySymbolsDontCare = result[1]
    if result[1] > maxNoisySymbolsDontCare:
        maxNoisySymbolsDontCare = result[1]
    resultsNoisySymbolsDontCare.append(result[1])

'''[
    noisyTrain,
    noisySymbolsTrain,
    targetsTrain,
    noisyValidation,
    noisySymbolsValidation,
    targetsValidation,
    noisyTest,
    noisySymbolsTest,
    targetsTest
] = dataLoader.getParallelData(4, 2, 2)

resultsNoisyParallel = []
minNoisyParallel = 1.0
maxNoisyParallel = 0.0
sumNoisyParallel = 0.0

for i in range(k):
    

    model = FeedForwardModel([512, 512], 1568, 20)
    model.train(noisyTrain, targetsTrain, noisyValidation, targetsValidation, 100, 200)
    result = model.evaluate(noisyTest, targetsTest)
    sumNoisyParallel += result[1]
    if result[1] < minNoisyParallel:
        minNoisyParallel = result[1]
    if result[1] > maxNoisyParallel:
        maxNoisyParallel = result[1]
    resultsNoisyParallel.append(result[1])

resultsNoisySymbolsParallel = []
minNoisySymbolsParallel = 1.0
maxNoisySymbolsParallel = 0.0
sumNoisySymbolsParallel = 0.0

for i in range(k):
    

    model = FeedForwardModel([512, 512], 1588, 20)
    model.train(noisySymbolsTrain, targetsTrain, noisySymbolsValidation, targetsValidation, 100, 200)
    result = model.evaluate(noisySymbolsTest, targetsTest)
    sumNoisySymbolsParallel += result[1]
    if result[1] < minNoisySymbolsParallel:
        minNoisySymbolsParallel = result[1]
    if result[1] > maxNoisySymbolsParallel:
        maxNoisySymbolsParallel = result[1]
    resultsNoisySymbolsParallel.append(result[1])

print("Parallel - Noisy")
print("RESULTS: " + str(resultsNoisyParallel))
print("SCORE: " + str(sumNoisyParallel / k))
print("MIN: " + str(minNoisyParallel))
print("MAX: " + str(maxNoisyParallel))

print("Parallel - Noisy and Symbols")
print("RESULTS: " + str(resultsNoisySymbolsParallel))
print("SCORE: " + str(sumNoisySymbolsParallel / k))
print("MIN: " + str(minNoisySymbolsParallel))
print("MAX: " + str(maxNoisySymbolsParallel))'''

'''print("Sequential with Don't Care - Noisy")
print("RESULTS: " + str(resultsNoisyDontCare))
print("SCORE: " + str(sumNoisyDontCare / k))
print("MIN: " + str(minNoisyDontCare))
print("MAX: " + str(maxNoisyDontCare))'''

print("Sequential with Don't Care - Noisy and Symbols")
print("RESULTS: " + str(resultsNoisySymbolsDontCare))
print("SCORE: " + str(sumNoisySymbolsDontCare / k))
print("MIN: " + str(minNoisySymbolsDontCare))
print("MAX: " + str(maxNoisySymbolsDontCare))

'''

Sequential - Noisy
RESULTS: [0.715, 0.715, 0.7, 0.68, 0.73]
SCORE: 0.708
MIN: 0.68
MAX: 0.73

Sequential - Noisy and Symbols
RESULTS: [0.875, 0.845, 0.885, 0.86, 0.83]
SCORE: 0.859
MIN: 0.83
MAX: 0.885

Sequential with Don't Care - Noisy
RESULTS: [0.71, 0.685, 0.71, 0.665, 0.69]
SCORE: 0.692
MIN: 0.665
MAX: 0.71

Sequential with Don't Care - Noisy and Symbols
RESULTS: [0.64, 0.61, 0.655, 0.67, 0.67]
SCORE: 0.649
MIN: 0.61
MAX: 0.67

Parallel - Noisy
RESULTS: [0.725, 0.72, 0.69, 0.68, 0.675]
SCORE: 0.6980000000000001
MIN: 0.675
MAX: 0.725

Parallel - Noisy and Symbols
RESULTS: [0.795, 0.77, 0.79, 0.76, 0.78]
SCORE: 0.7790000000000001
MIN: 0.76
MAX: 0.795

'''