from __future__ import division, print_function, absolute_import
from utils import argmax, getOperator, renderImage, renderDigit
from lstm_model import LSTMModel
import numpy as np
from random import randint
from PIL import Image, ImageDraw, ImageFont

def one_hot(value):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

plus_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
equals_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

x = []
y = []

print(one_hot((5 + 6) % 10) + [((5 + 6) % 10) / 10.0])

for a in range(10):
    for b in range(10):
        x.append([one_hot(b), one_hot(a), plus_vector, equals_vector])
        y.append([one_hot(b) + [b / 10.0], one_hot(a) + [a / 10.0], one_hot((a + b) % 10) + [((a + b) % 10) / 10.0], one_hot((a + b) // 10) + [((a + b) // 10) / 10.0]])

x = np.array(x)
y = np.array(y)

model = LSTMModel(4, [10, 10], 12, 13)
model.train(x, y, x, y, 10, 5000)

results = model.predict(x)
activations_0 = model.getActivations(0, x)
activations_1 = model.getActivations(1, x)

count = 0

with open("ordinal.txt", "w+") as f:
    for index, result in enumerate(results):
        left = argmax(result[2][0:13])
        right = argmax(result[3][0:13])
        leftTarget = argmax(y[index][2][0:13])
        rightTarget = argmax(y[index][3][0:13])

        if left == leftTarget and right == rightTarget:
            count += 1

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
        f.write("\n\n")


renderResults("symbols-activations-ordinal", results, x, activations_0, activations_1)

print("SCORE: " + str(float(count) / len(results)))