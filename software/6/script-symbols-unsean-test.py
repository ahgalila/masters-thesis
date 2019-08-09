from __future__ import division, print_function, absolute_import
from utils import argmax, getOperator, renderImage, renderDigit
from lstm_model import LSTMModel
import numpy as np
from random import randint
from PIL import Image, ImageDraw, ImageFont
import sys

def one_hot(value):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result[value] = 1
    return result

def renderActivations(draw, activations, startX, startY):
    for index in range(10):
        intensity = int(((activations[index] + 1) / 2.0) * 255)
        for x in range(startX + index * 14, startX + index * 14 + 14):
            for y in range(startY, startY + 14):
                draw.point((x, y), fill=(intensity, intensity, intensity))

def renderResults(name, results, test_x, activations_0, activations_1):
    
    image = Image.new('RGB', (28 * 3 + 280, len(results) * 224))
    draw = ImageDraw.Draw(image)

    for index, result in enumerate(results):

        renderImage(image, "MNIST_data/" + str(argmax(test_x[index][0])) + ".png", 0 * 28, index * 224 + 0 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[0][:-1])) + ".png", 2 * 28, index * 224 + 0 * 32)
        renderActivations(draw, activations_1[index][0], 3 * 28, index * 224 + 0 * 32)
        renderActivations(draw, activations_0[index][0], 3 * 28, index * 224 + 0 * 32 + 14)

        renderImage(image, "MNIST_data/" + str(argmax(test_x[index][1])) + ".png", 0 * 28, index * 224 + 1 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[1][:-1])) + ".png", 2 * 28, index * 224 + 1 * 32)
        renderActivations(draw, activations_1[index][1], 3 * 28, index * 224 + 1 * 32)
        renderActivations(draw, activations_0[index][1], 3 * 28, index * 224 + 1 * 32 + 14)

        renderImage(image, "MNIST_data/" + "plus.png", 0 * 28, index * 224 + 2 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[2][:-1])) + ".png", 2 * 28, index * 224 + 2 * 32)
        renderActivations(draw, activations_1[index][2], 3 * 28, index * 224 + 2 * 32)
        renderActivations(draw, activations_0[index][2], 3 * 28, index * 224 + 2 * 32 + 14)

        renderImage(image, "MNIST_data/" + "equals.png", 0 * 28, index * 224 + 3 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[3][:-1])) + ".png", 2 * 28, index * 224 + 3 * 32)
        renderActivations(draw, activations_1[index][3], 3 * 28, index * 224 + 3 * 32)
        renderActivations(draw, activations_0[index][3], 3 * 28, index * 224 + 3 * 32 + 14)

    image.save(name  + ".png")

empty = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plus_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
equals_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

x = []
y = []

test_x = []
test_y = []

testWatchListA = {}
testWatchListB = {}

for a in range(10):
    testWatchListA[str(a)] = randint(0, 9)
for b in range(10):
    testWatchListB[str(b)] = randint(0, 9)

for a in range(10):
    for b in range(10):
        if b == testWatchListA[str(a)] or a == testWatchListB[str(b)]:
            #test_x.append([empty + [b / 10.0, 1, 0, 0], empty + [a / 10.0, 1, 0, 0], plus_vector, equals_vector])
            #test_y.append([empty + [b / 10.0], empty + [a / 10.0], empty + [((a + b) % 10) / 10.0], empty + [((a + b) // 10) / 10.0]])
            test_x.append([one_hot(b) + [b / 10.0, 1, 0, 0], one_hot(a) + [a / 10.0, 1, 0, 0], plus_vector, equals_vector])
            test_y.append([one_hot(b) + [b / 10.0], one_hot(a) + [a / 10.0], one_hot((a + b) % 10) + [((a + b) % 10) / 10.0], one_hot((a + b) // 10) + [((a + b) // 10) / 10.0]])
        else:
            print(str(a) + " ::: " + str(b))
            #x.append([empty + [b / 10.0, 1, 0, 0], empty + [a / 10.0, 1, 0, 0], plus_vector, equals_vector])
            #y.append([empty + [b / 10.0], empty + [a / 10.0], empty + [((a + b) % 10) / 10.0], empty + [((a + b) // 10) / 10.0]])
            x.append([one_hot(b) + [b / 10.0, 1, 0, 0], one_hot(a) + [a / 10.0, 1, 0, 0], plus_vector, equals_vector])
            y.append([one_hot(b) + [b / 10.0], one_hot(a) + [a / 10.0], one_hot((a + b) % 10) + [((a + b) % 10) / 10.0], one_hot((a + b) // 10) + [((a + b) // 10) / 10.0]])

sys.exit(0)

x = np.array(x)
y = np.array(y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [100, 100], 14, 11)
model.train(x, y, test_x, test_y, 10, 5000)

results = model.predict(x)
activations_0 = model.getActivations(0, x)
activations_1 = model.getActivations(1, x)

count = 0
count_ordinal = 0

for index, result in enumerate(results):
    left = argmax(result[2][:-1])
    right = argmax(result[3][:-1])
    leftTarget = argmax(y[index][2][:-1])
    rightTarget = argmax(y[index][3][:-1])

    if left == leftTarget and right == rightTarget:
        count += 1

    leftOrdinal = round(result[2][-1], 1)
    rightOrdinal = round(result[3][-1], 1)
    leftTargetOrdinal = round(y[index][2][-1], 1)
    rightTargetOrdinal = round(y[index][3][-1], 1)

    if leftOrdinal == leftTargetOrdinal and rightOrdinal == rightTargetOrdinal:
        count_ordinal += 1

test_count = 0
test_count_ordinal = 0

test_results = model.predict(test_x)
with open("ordinal-unseen-3.txt", "w+") as f:
    for index, result in enumerate(test_results):
        left = argmax(result[2][:-1])
        right = argmax(result[3][:-1])
        leftTarget = argmax(test_y[index][2][:-1])
        rightTarget = argmax(test_y[index][3][:-1])

        if left == leftTarget and right == rightTarget:
            test_count += 1

        leftOrdinal = round(result[2][-1], 1)
        rightOrdinal = round(result[3][-1], 1)
        leftTargetOrdinal = round(test_y[index][2][-1], 1)
        rightTargetOrdinal = round(test_y[index][3][-1], 1)

        if leftOrdinal == leftTargetOrdinal and rightOrdinal == rightTargetOrdinal:
            test_count_ordinal += 1

        f.write(str(test_x[index][0][-1]) + " => " + str(result[0][-1]) + "\n")
        f.write(str(test_x[index][1][-1]) + " => " + str(result[1][-1]) + "\n")
        f.write("+" + " => " + str(result[2][-1]) + "\n")
        f.write("=" + " => " + str(result[3][-1]) + "\n")
        f.write("\n\n")

'''with open("ordinal-unseen-2.txt", "w+") as f:
    for index, result in enumerate(test_results):
        left = argmax(result[2][:-1])
        right = argmax(result[3][:-1])
        leftTarget = argmax(test_y[index][2][:-1])
        rightTarget = argmax(test_y[index][3][:-1])

        if left == leftTarget and right == rightTarget:
            test_count += 1

        b = argmax(x[index][0][0:13])
        bOrdinal = result[0][-1]
        a = argmax(x[index][1][0:13])
        aOrdinal = result[1][-1]
        leftOrdinal = result[2][-1]
        rightOrdinal = result[3][-1]

        f.write(str(b) + " => " + str(bOrdinal) + "\n")
        f.write(str(a) + " => " + str(aOrdinal) + "\n")
        f.write(str(left) + " => " + str(leftOrdinal) + "\n")
        f.write(str(right) + " => " + str(rightOrdinal) + "\n")
        f.write("\n\n")'''

#renderResults("symbols-activations-ordinal", results, x, activations_0, activations_1)

print("SCORE: " + str(float(count) / len(results)))
print("TEST SCORE: " + str(float(test_count) / len(test_results)))
print("SCORE ORDINAL: " + str(float(count_ordinal) / len(results)))
print("TEST SCORE ORDINAL: " + str(float(test_count_ordinal) / len(test_results)))