from __future__ import division, print_function, absolute_import
from utils import argmax, one_hot, getOperator, renderImage, renderDigit
from lstm_model import LSTMModel
import numpy as np
from random import randint
from PIL import Image, ImageDraw, ImageFont

def renderActivations(draw, activations, startX, startY):
    for index in range(20):
        intensity = int(((activations[index] + 1) / 2.0) * 255)
        for x in range(startX + index * 14, startX + index * 14 + 14):
            for y in range(startY, startY + 14):
                draw.point((x, y), fill=(intensity, intensity, intensity))

def renderResults(name, results, test_x, activations_0, activations_1):
    
    image = Image.new('RGB', (28 * 3 + 280, len(results) * 160))
    draw = ImageDraw.Draw(image)

    for index, result in enumerate(results):

        renderImage(image, "MNIST_data/" + str(argmax(test_x[index][0])) + ".png", 0 * 28, index * 160 + 0 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[0][10:])) + ".png", 2 * 28, index * 160 + 0 * 32)
        renderActivations(draw, activations_1[index][0], 3 * 28, index * 160 + 0 * 32)
        renderActivations(draw, activations_0[index][0], 3 * 28, index * 160 + 0 * 32 + 14)

        renderImage(image, "MNIST_data/" + str(argmax(test_x[index][1])) + ".png", 0 * 28, index * 160 + 1 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[1][10:])) + ".png", 2 * 28, index * 160 + 1 * 32)
        renderActivations(draw, activations_1[index][1], 3 * 28, index * 160 + 1 * 32)
        renderActivations(draw, activations_0[index][1], 3 * 28, index * 160 + 1 * 32 + 14)

        renderImage(image, "MNIST_data/" + getOperator(test_x[index][2][:4]) + ".png", 0 * 28, index * 160 + 2 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[2][:10])) + ".png", 1 * 28, index * 160 + 2 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[2][10:20])) + ".png", 2 * 28, index * 160 + 2 * 32)
        renderActivations(draw, activations_1[index][2], 3 * 28, index * 160 + 2 * 32)
        renderActivations(draw, activations_0[index][2], 3 * 28, index * 160 + 2 * 32 + 14)

    image.save(name  + ".png")

plus_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
            test_x.append([one_hot(b), one_hot(a), plus_vector])
            test_y.append([one_hot(0) +  one_hot(b), one_hot(0) +  one_hot(a), one_hot((a + b) // 10) + one_hot((a + b) % 10)])
        else:
            x.append([one_hot(b), one_hot(a), plus_vector])
            y.append([one_hot(0) +  one_hot(b), one_hot(0) +  one_hot(a), one_hot((a + b) // 10) + one_hot((a + b) % 10)])

x = np.array(x)
y = np.array(y)

test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(3, [20, 20], 10, 20)
model.train(x, y, x, y, 10, 5000)

results = model.predict(x)
activations_0 = model.getActivations(0, x)
activations_1 = model.getActivations(1, x)

count = 0

for index, result in enumerate(results):
    left = argmax(result[2][-20:-10])
    right = argmax(result[2][-10:])
    leftTarget = argmax(y[index][2][-20:-10])
    rightTarget = argmax(y[index][2][-10:])

    if left == leftTarget and right == rightTarget:
        count += 1

test_count = 0

test_results = model.predict(test_x)
for index, result in enumerate(test_results):
    left = argmax(result[2][-20:-10])
    right = argmax(result[2][-10:])
    leftTarget = argmax(test_y[index][2][-20:-10])
    rightTarget = argmax(test_y[index][2][-10:])

    if left == leftTarget and right == rightTarget:
        test_count += 1

#renderResults("symbols-activations", results, x, activations_0, activations_1)

print("SCORE: " + str(float(count) / len(results)))
print("TEST SCORE: " + str(float(test_count) / len(test_results)))