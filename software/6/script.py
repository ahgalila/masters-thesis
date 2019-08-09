from __future__ import division, print_function, absolute_import
import utils
import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from data import DataLoader
from test import TestDataLoader
from utils import argmax, renderImage, renderDigit, getOperator
from PIL import Image, ImageDraw, ImageFont

dataLoader = DataLoader(src="MNIST_data/dataset")
testDataLoader = TestDataLoader(src="MNIST_data/dataset")

num_with_symbols = 0

k = 5
tctoScores = []
tcfoScores = []
fctoScores = []
fcfoScores = []
epochs = []

k_fold_index = 0
while k_fold_index < k:

    [
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y
    ] = dataLoader.getData(k_fold_index, 4, 4, num_with_symbols)

    model = LSTMModel(4, [512, 512], 784, 20)
    epoch = model.train(train_x, train_y, val_x, val_y, 10, 200)
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

        leftClass = argmax(result[1][-10:])
        rightClass = argmax(result[0][-10:])
        leftClassTarget = (leftClass + rightClass) // 10
        rightClassTarget = (leftClass + rightClass) % 10

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


    k_fold_index += 1

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

'''count = 0
total = 0

classesCount = 0
classesErrorCount = 0

[
    test_x,
    test_y
] = testDataLoader.getData()

results = bestModel.predict(test_x)

image = Image.new('RGB', (28 * 10, 28 * len(results)))
draw = ImageDraw.Draw(image)

for index, result in enumerate(results):
            
    left = argmax(result[3][-20:-10])
    right = argmax(result[3][-10:])
    leftTarget = argmax(test_y[index][3][-20:-10])
    rightTarget = argmax(test_y[index][3][-10:])

    leftClass = argmax(result[1][-10:])
    rightClass = argmax(result[0][-10:])
    leftClassTarget = (leftClass + rightClass) // 10
    rightClassTarget = (leftClass + rightClass) % 10

    if left == leftTarget and right == rightTarget:
        count += 1
    else:
        if leftClassTarget == leftTarget and rightClassTarget == rightTarget:
            classesErrorCount += 1
    if leftClassTarget == leftTarget and rightClassTarget == rightTarget:
        classesCount += 1
    total += 1

    renderDigit(draw, test_x[index][1], 0 * 28, index * 28)
    renderDigit(draw, test_x[index][2], 1 * 28, index * 28)
    renderDigit(draw, test_x[index][0], 2 * 28, index * 28)
    renderImage(image, "MNIST_data/implies.png", 3 * 28, index * 28)
    renderImage(image, "MNIST_data/" + str(argmax(result[1][-10:])) + ".png", 4 * 28, index * 28)
    renderImage(image, "MNIST_data/" + getOperator(result[2][:4]) + ".png", 5 * 28, index * 28)
    renderImage(image, "MNIST_data/" + str(argmax(result[0][-10:])) + ".png", 6 * 28, index * 28)
    renderImage(image, "MNIST_data/equals.png", 7 * 28, index * 28)
    renderImage(image, "MNIST_data/" + str(argmax(result[3][-20:-10])) + ".png", 8 * 28, index * 28)
    renderImage(image, "MNIST_data/" + str(argmax(result[3][-10:])) + ".png", 9 * 28, index * 28)

#image.save("results-" + str(num_with_symbols) + ".png")

print("SCORES: " + str(scores))
print("EPOCHS: " + str(epochs))
print("CLASS SCORES: " + str(classScores))
print("CLASS ERROR SCORES: " + str(classErrorScores))

print("SCORE: " + str(count / float(total)))
print("CLASS SCORE: " + str(classesCount / float(total)))
print("CLASS ERROR SCORE: " + str(classesErrorCount / float(total - count)))'''


'''
100% Symbols
----------------------
TCTO: 
[0.755, 0.765, 0.7775, 0.775, 0.7175]
TCfO: 
[0.0925, 0.0825, 0.08, 0.0525, 0.1]
fCTO: 
[0.025, 0.0175, 0.0225, 0.03, 0.0275]
fCfO: 
[0.1275, 0.135, 0.12, 0.1425, 0.155]
EPOCHS: 
[71, 22, 22, 51, 24]
'''

'''
75% Symbols
----------------------
TCTO: 
[0.6175, 0.6875, 0.6475, 0.65, 0.5975]
TCfO: 
[0.1525, 0.14, 0.1375, 0.125, 0.1225]
fCTO: 
[0.045, 0.04, 0.045, 0.0625, 0.0775]
fCfO: 
[0.185, 0.1325, 0.17, 0.1625, 0.2025]
EPOCHS: 
[15, 25, 13, 17, 9]
'''

'''
50%
----------------------
TCTO: 
[0.455, 0.255, 0.48, 0.4425, 0.3225]
TCfO: 
[0.1525, 0.4325, 0.175, 0.2375, 0.1625]
fCTO: 
[0.1125, 0.0675, 0.1175, 0.1, 0.1775]
fCfO: 
[0.28, 0.245, 0.2275, 0.22, 0.3375]
EPOCHS: 
[21, 1, 14, 8, 5]
'''

'''
25% symbols
------------------
TCTO: 
[0.16, 0.15, 0.1925, 0.18, 0.1775]
TCfO: 
[0.13, 0.1825, 0.1475, 0.125, 0.1675]
fCTO: 
[0.3225, 0.235, 0.315, 0.295, 0.23]
fCfO: 
[0.3875, 0.4325, 0.345, 0.4, 0.425]
EPOCHS: 
[13, 9, 14, 9, 5]
'''

'''
0% symbols
-------------------------
TCTO: 
[0.0125, 0.06, 0.035, 0.0225, 0.0175]
TCfO: 
[0.05, 0.03, 0.025, 0.05, 0.03]
fCTO: 
[0.325, 0.365, 0.49, 0.2925, 0.405]
fCfO: 
[0.6125, 0.545, 0.45, 0.635, 0.5475]
EPOCHS: 
[3, 4, 8, 2, 4]
'''

'''
100% symbols
[0.7625, 0.7475, 0.7675, 0.815, 0.78]
SCORE: 0.803
'''

'''
0% symbols
--------------------
SCORES: [0.3525, 0.225, 0.315, 0.33, 0.3625]
EPOCHS: [3, 0, 1, 7, 5]
CLASS SCORES: [0.07, 0.0675, 0.0725, 0.0775, 0.075]
CLASS ERROR SCORES: [0.07722007722007722, 0.06129032258064516, 0.07664233576642336, 0.08582089552238806, 0.07450980392156863]
SCORE: 0.389
CLASS SCORE: 0.087
CLASS ERROR SCORE: 0.0981996726678
'''

'''
25% symbols
----------------------
SCORES: [0.4975, 0.2125, 0.39, 0.4725, 0.4475]
EPOCHS: [11, 2, 8, 13, 9]
CLASS SCORES: [0.3125, 0.3775, 0.4225, 0.325, 0.3225]
CLASS ERROR SCORES: [0.2885572139303483, 0.3523809523809524, 0.36885245901639346, 0.2796208530805687, 0.28054298642533937]
SCORE: 0.475
CLASS SCORE: 0.277
CLASS ERROR SCORE: 0.245714285714
'''

'''
50% symbols
----------------------
SCORES: [0.3675, 0.4975, 0.425, 0.5, 0.4925]
EPOCHS: [2, 11, 4, 5, 8]
CLASS SCORES: [0.4675, 0.6475, 0.5825, 0.6525, 0.7025]
CLASS ERROR SCORES: [0.3715415019762846, 0.5024875621890548, 0.47391304347826085, 0.56, 0.5566502463054187]
SCORE: 0.489
CLASS SCORE: 0.621
CLASS ERROR SCORE: 0.493150684932
'''

'''
75% symbols
----------------------
SCORES: [0.6475, 0.675, 0.62, 0.7, 0.6475]
EPOCHS: [9, 19, 7, 5, 6]
CLASS SCORES: [0.745, 0.7725, 0.7575, 0.78, 0.7675]
CLASS ERROR SCORES: [0.49645390070921985, 0.47692307692307695, 0.5263157894736842, 0.475, 0.5319148936170213]
SCORE: 0.677
CLASS SCORE: 0.791
CLASS ERROR SCORE: 0.482972136223
'''

'''
100% symbols
----------------------
SCORES: [0.77, 0.73, 0.7575, 0.785, 0.8025]
EPOCHS: [15, 19, 19, 22, 105]
CLASS SCORES: [0.8225, 0.825, 0.8325, 0.87, 0.8525]
CLASS ERROR SCORES: [0.34782608695652173, 0.4166666666666667, 0.3711340206185567, 0.5, 0.379746835443038]
SCORE: 0.791
CLASS SCORE: 0.851
CLASS ERROR SCORE: 0.392344497608
'''