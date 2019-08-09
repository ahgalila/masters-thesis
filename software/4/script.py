from __future__ import division, print_function, absolute_import
import utils
import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from data import DataLoader
from utils import argmax, renderImage, renderDigit, getOperator
from PIL import Image, ImageDraw, ImageFont

dataLoader = DataLoader()

k = 1
k_fold_index = 0
scores = []
scores_plus = []
scores_times = []
scores_minus = []
scores_divide = []

while k_fold_index < k:

    [train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y] = dataLoader.getDataWithClasses(k_fold_index, 2, 2)

    print(len(train_x))
    print(len(val_x))
    print(len(test_x))

    model = LSTMModel(4, [512, 512, 512], 784, 20)
    model.train(train_x, train_y, val_x, val_y, 100, 200)

    count = 0
    total = 0

    results = model.predict(test_x)

    image = Image.new('RGB', (28 * 10, 28 * len(results)))
    draw = ImageDraw.Draw(image)

    for index, result in enumerate(results):
        left = argmax(result[3][-20:-10])
        right = argmax(result[3][-10:])
        leftTarget = argmax(test_y[index][3][-20:-10])
        rightTarget = argmax(test_y[index][3][-10:])

        if left == leftTarget and right == rightTarget:
            count += 1
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

    image.save("results.png")
    scores.append(count / float(total))

    '''count = 0
    total = 0
    results = model.predict(test_x[0:200])
    for index, result in enumerate(results):
        left = argmax(result[3][-20:-10])
        right = argmax(result[3][-10:])
        leftTarget = argmax(test_y[index][3][-20:-10])
        rightTarget = argmax(test_y[index][3][-10:])

        if left == leftTarget and right == rightTarget:
            count += 1
        total += 1
    scores_plus.append(count / float(total))

    count = 0
    total = 0
    results = model.predict(test_x[200:400])
    for index, result in enumerate(results):
        left = argmax(result[3][-20:-10])
        right = argmax(result[3][-10:])
        leftTarget = argmax(test_y[index][3][-20:-10])
        rightTarget = argmax(test_y[index][3][-10:])

        if left == leftTarget and right == rightTarget:
            count += 1
        total += 1
    scores_times.append(count / float(total))

    count = 0
    total = 0
    results = model.predict(test_x[400:510])
    for index, result in enumerate(results):
        left = argmax(result[3][-20:-10])
        right = argmax(result[3][-10:])
        leftTarget = argmax(test_y[index][3][-20:-10])
        rightTarget = argmax(test_y[index][3][-10:])

        if left == leftTarget and right == rightTarget:
            count += 1
        total += 1
    scores_minus.append(count / float(total))

    count = 0
    total = 0
    results = model.predict(test_x[510:])
    for index, result in enumerate(results):
        left = argmax(result[3][-20:-10])
        right = argmax(result[3][-10:])
        leftTarget = argmax(test_y[index][3][-20:-10])
        rightTarget = argmax(test_y[index][3][-10:])

        if left == leftTarget and right == rightTarget:
            count += 1
        total += 1
    scores_divide.append(count / float(total))'''

    k_fold_index += 1

print(scores)
'''print(scores_plus)
print(scores_times)
print(scores_minus)
print(scores_divide)'''