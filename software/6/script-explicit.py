from __future__ import division, print_function, absolute_import
import utils
import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from data_explicit import DataLoader
from test import TestDataLoader
from utils import argmax, renderImage, renderDigit, getOperator, getOperatorSign
from PIL import Image, ImageDraw, ImageFont

def renderResults(name, results, test_y):
    
    image = Image.new('RGB', (28 * 14, 28 * len(results)))
    draw = ImageDraw.Draw(image)

    with open(name  + ".txt", "w") as f:

        for index, result in enumerate(results):

            strIndex = "{:03}".format(index)

            f.write(strIndex + ": " + str(argmax(test_y[index][3][10:20])) + " " + getOperatorSign(test_y[index][3][20:30]) + " " + str(argmax(test_y[index][3][:10])) + "\r\n")
            f.write("\t" + str(argmax(result[0][:10])) + " " + str(argmax(result[0][10:20])) + " " + getOperatorSign(result[0][20:30]) + " " + str(argmax(result[0][-20:-10])) + " " + str(argmax(result[0][-10:])) + "\r\n")
            f.write("\t" + str(argmax(result[1][:10])) + " " + str(argmax(result[1][10:20])) + " " + getOperatorSign(result[1][20:30]) + " " + str(argmax(result[1][-20:-10])) + " " + str(argmax(result[1][-10:])) + "\r\n")
            f.write("\t" + str(argmax(result[2][:10])) + " " + str(argmax(result[2][10:20])) + " " + getOperatorSign(result[2][20:30]) + " " + str(argmax(result[2][-20:-10])) + " " + str(argmax(result[2][-10:])) + "\r\n")
            f.write("\t" + str(argmax(result[3][:10])) + " " + str(argmax(result[0][10:20])) + " " + getOperatorSign(result[3][20:30]) + " " + str(argmax(result[3][-20:-10])) + " " + str(argmax(result[3][-10:])) + "\r\n")

            renderImage(image, "MNIST_data/" + strIndex[:1] + ".png", 0 * 28, index * 28)
            renderImage(image, "MNIST_data/" + strIndex[1:2] + ".png", 1 * 28, index * 28)
            renderImage(image, "MNIST_data/" + strIndex[2:] + ".png", 2 * 28, index * 28)
            renderImage(image, "MNIST_data/colon.png", 3 * 28, index * 28)
            renderDigit(draw, test_x[index][1], 4 * 28, index * 28)
            renderDigit(draw, test_x[index][2], 5 * 28, index * 28)
            renderDigit(draw, test_x[index][0], 6 * 28, index * 28)
            renderImage(image, "MNIST_data/implies.png", 7 * 28, index * 28)
            renderImage(image, "MNIST_data/" + str(argmax(result[3][10:20])) + ".png", 8 * 28, index * 28)
            renderImage(image, "MNIST_data/" + getOperator(result[3][20:24]) + ".png", 9 * 28, index * 28)
            renderImage(image, "MNIST_data/" + str(argmax(result[3][:10])) + ".png", 10 * 28, index * 28)
            renderImage(image, "MNIST_data/equals.png", 11 * 28, index * 28)
            renderImage(image, "MNIST_data/" + str(argmax(result[3][-20:-10])) + ".png", 12 * 28, index * 28)
            renderImage(image, "MNIST_data/" + str(argmax(result[3][-10:])) + ".png", 13 * 28, index * 28)

        image.save(name  + ".png")

dataLoader = DataLoader(src="MNIST_data/dataset")

num_with_symbols = 0

k = 5
tctoScores = []
tcfoScores = []
fctoScores = []
fcfoScores = []
epochs = []

k_fold_index = 0
bestModel = None
maxAcccuracy = 0
while k_fold_index < k:

    [
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y
    ] = dataLoader.getData(k_fold_index, 4, 4, num_with_symbols)

    model = LSTMModel(4, [512, 512], 784, 50)
    epoch = model.train(train_x, train_y, val_x, val_y, 100, 200)
    epochs.append(epoch)

    results = model.predict(test_x)

    tcto = 0
    tcfo = 0
    fcto = 0
    fcfo = 0
    total = 0

    for index, result in enumerate(results):
            
        left = argmax(result[3][-20:-10])
        right = argmax(result[3][-10:])
        leftTarget = argmax(test_y[index][3][-20:-10])
        rightTarget = argmax(test_y[index][3][-10:])

        leftClass = argmax(result[3][10:20])
        rightClass = argmax(result[3][:10])
        operator = argmax(test_y[index][3][20:30])
        if operator == 0:
            leftClassTarget = (leftClass + rightClass) // 10
            rightClassTarget = (leftClass + rightClass) % 10
        if operator == 1:
            leftClassTarget = (leftClass * rightClass) // 10
            rightClassTarget = (leftClass * rightClass) % 10
        if operator == 2:
            leftClassTarget = (leftClass - rightClass) // 10
            rightClassTarget = (leftClass - rightClass) % 10
        if operator == 3 and rightClass > 0:
            leftClassTarget = leftClass // rightClass
            rightClassTarget = leftClass % rightClass

        if left == leftTarget and right == rightTarget:
            if leftClassTarget == leftTarget and rightClassTarget == rightTarget:
                tcto += 1
            else:
                fcto += 1
        else:
            if leftClassTarget == leftTarget and rightClassTarget == rightTarget:
                tcfo += 1
            else:
                fcfo += 1
        total += 1

    tctoScores.append(tcto / float(total))
    tcfoScores.append(tcfo / float(total))
    fctoScores.append(fcto / float(total))
    fcfoScores.append(fcfo / float(total))

    if fcto + tcto > maxAcccuracy:
        maxAcccuracy = fcto + tcto
        bestModel = model

    k_fold_index += 1

testDataLoader = TestDataLoader(src="MNIST_data/dataset")

#plus
[test_x, test_y] = testDataLoader.getExplicitData("+")
results = bestModel.predict(test_x)
renderResults("results-addition-" + str(num_with_symbols), results, test_y)

#times
[test_x, test_y] = testDataLoader.getExplicitData("x")
results = bestModel.predict(test_x)
renderResults("results-multiplication-" + str(num_with_symbols), results, test_y)

#minus
[test_x, test_y] = testDataLoader.getExplicitData("-")
results = bestModel.predict(test_x)
renderResults("results-subtraction-" + str(num_with_symbols), results, test_y)

#divide
[test_x, test_y] = testDataLoader.getExplicitData("/")
results = bestModel.predict(test_x)
renderResults("results-division-" + str(num_with_symbols), results, test_y)

print("TCTO: ")
print(tctoScores)
print("TCfO: ")
print(tcfoScores)
print("fCTO: ")
print(fctoScores)
print("fCfO: ")
print(fcfoScores)
print("EPOCHS: ")
print(epochs)

'''
100% symbols
------------
TCTO:
[0.7652173913043478, 0.7594202898550725, 0.7847826086956522, 0.7710144927536232, 0.7942028985507247]
TCfO: 
[0.04710144927536232, 0.057971014492753624, 0.06739130434782609, 0.05144927536231884, 0.04782608695652174]
fCTO: 
[0.018115942028985508, 0.028985507246376812, 0.028985507246376812, 0.018115942028985508, 0.02463768115942029]
fCfO: 
[0.16956521739130434, 0.1536231884057971, 0.11884057971014493, 0.15942028985507245, 0.13333333333333333]
EPOCHS: 
[77, 83, 104, 64, 50]
'''

'''
75% symbols
-----------
TCTO: 
[0.572463768115942, 0.6086956521739131, 0.5840579710144927, 0.6202898550724638, 0.5673913043478261]
TCfO: 
[0.16956521739130434, 0.12173913043478261, 0.15507246376811595, 0.12173913043478261, 0.15579710144927536]
fCTO: 
[0.050724637681159424, 0.08260869565217391, 0.06159420289855073, 0.06884057971014493, 0.06884057971014493]
fCfO: 
[0.2072463768115942, 0.18695652173913044, 0.19927536231884058, 0.1891304347826087, 0.20797101449275363]
EPOCHS: 
[8, 44, 7, 25, 8]
'''

'''
50% symbols
-----------
[0.29782608695652174, 0.28043478260869564, 0.3615942028985507, 0.2710144927536232, 0.4072463768115942]
TCfO: 
[0.21811594202898552, 0.2217391304347826, 0.19855072463768117, 0.25869565217391305, 0.1427536231884058]
fCTO: 
[0.10507246376811594, 0.1427536231884058, 0.11884057971014493, 0.09420289855072464, 0.17681159420289855]
fCfO: 
[0.3789855072463768, 0.35507246376811596, 0.3210144927536232, 0.3760869565217391, 0.27318840579710146]
EPOCHS: 
[6, 6, 7, 5, 14]
'''

'''
25% symbols
-----------
TCTO: 
[0.13260869565217392, 0.11666666666666667, 0.11811594202898551, 0.14492753623188406, 0.10942028985507246]
TCfO: 
[0.12173913043478261, 0.1492753623188406, 0.18333333333333332, 0.1282608695652174, 0.1391304347826087]
fCTO: 
[0.2514492753623188, 0.16956521739130434, 0.1398550724637681, 0.22391304347826088, 0.17101449275362318]
fCfO: 
[0.49420289855072463, 0.5644927536231884, 0.558695652173913, 0.5028985507246376, 0.5804347826086956]
EPOCHS: 
[7, 4, 2, 7, 3]
'''

'''
0% symbols
----------
TCTO: 
[0.04492753623188406, 0.01956521739130435, 0.06376811594202898, 0.05, 0.04855072463768116]
TCfO: 
[0.021014492753623187, 0.018115942028985508, 0.015942028985507246, 0.01956521739130435, 0.014492753623188406]
fCTO: 
[0.5688405797101449, 0.5492753623188406, 0.5717391304347826, 0.55, 0.5920289855072464]
fCfO: 
[0.3652173913043478, 0.41304347826086957, 0.34855072463768116, 0.3804347826086957, 0.34492753623188405]
EPOCHS: 
[26, 29, 47, 25, 56]
'''