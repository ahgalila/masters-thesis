from __future__ import division, print_function, absolute_import
from data import DataLoader
from lstm_model import LSTMModel
import utils


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
] = dataLoader.getSequencialDataWithDontCare(0, 2, 2)

model = LSTMModel(2, [512, 512, 512], 1568, 20)
model.train(noisyTrain, targetsTrain, noisyValidation, targetsValidation, 100, 100)
result = model.evaluate(noisyTest, targetsTest)
predictions = model.predict(noisyTest)
count = 0
for index in range(len(predictions)):
    a = utils.argmax(predictions[index][:10])
    b = utils.argmax(predictions[index][10:])
    predicted = a * 10 + b

    a = utils.argmax(targetsTest[index][:10])
    b = utils.argmax(targetsTest[index][10:])
    actual = a * 10 + b

    if actual == predicted:
        count += 1


print("ACCURACY: " + str(result[1]))
print("SCORE: " + str(count / 200.0))

utils.renderResults(noisyTest, predictions)