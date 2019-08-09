from __future__ import division, print_function, absolute_import
import utils

import sys
sys.dont_write_bytecode = True

from data import DataLoader
from lstm_model import LSTMModel

k = 1
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
    ] = dataLoader.getSequencialDataNoClassification(i, 2, 2)

    model = LSTMModel(3, [512], 784, 20)
    model.train(noisyTrain, targetsTrain, noisyValidation, targetsValidation, 100, 50)
    result = model.evaluate(noisyTest, targetsTest)
    #predictions = model.predict(noisyTest)

    #utils.renderResults("results_ideal" + str(i) + ".png", noisyTest, predictions)

    results.append(result[1])
    total += result[1]

print(results)
print("SCORE: " + str(total / k))

#No Classification
#[0.26166666626930235, 0.4266666686534882, 0.4099999964237213, 0.2849999988079071, 0.28833332896232605]
#SCORE: 0.33433333182334896

#Classification
#[0.821666669845581, 0.831666669845581, 0.8766666674613952, 0.846666669845581, 0.8333333325386048]
#SCORE: 0.8420000019073486