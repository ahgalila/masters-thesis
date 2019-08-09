from __future__ import division, print_function, absolute_import
import utils
import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from utils import argmax, one_hot, argmax, renderImage, renderDigit, getOperator
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageDraw, ImageFont
from random import randint

plus = np.load("MNIST_data/plus.npy")
minus = np.load("MNIST_data/minus.npy")
times = np.load("MNIST_data/times.npy")
divide = np.load("MNIST_data/divide.npy")
equals = np.load("MNIST_data/equals.npy")

empty_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
plus_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
times_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
minus_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
divide_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dontCare = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
empty_operation = [0, 0, 0, 0]

sequancesWithClasses = []
sequancesWithoutClasses = []

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainSamples = [[], [], [], [], [], [], [], [], [], []]
testSamples = [[], [], [], [], [], [], [], [], [], []]

for i in range(len(mnist.train.images)):
    trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])
for i in range(len(mnist.test.images)):
    testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])

for a in range(0, 10):
    for b in range(0, 10):

        for i in range(10):
            left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
            right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

            #plus
            target = [empty_one_hot + one_hot(b), empty_one_hot + one_hot(a), plus_vector, one_hot((a + b) // 10) + one_hot((a + b) % 10)]
            sequancesWithClasses.append(([right, left, plus, equals], target))
            target = [dontCare, dontCare, dontCare, one_hot((a + b) // 10) + one_hot((a + b) % 10)]
            sequancesWithoutClasses.append(([right, left, plus, equals], target))

            #times
            target = [empty_one_hot + one_hot(b), empty_one_hot + one_hot(a), times_vector, one_hot((a * b) // 10) + one_hot((a * b) % 10)]
            sequancesWithClasses.append(([right, left, times, equals], target))
            target = [dontCare, dontCare, dontCare, one_hot((a * b) // 10) + one_hot((a * b) % 10)]
            sequancesWithoutClasses.append(([right, left, times, equals], target))

            #minus
            if a >= b:
                target = [empty_one_hot + one_hot(b), empty_one_hot + one_hot(a), minus_vector, one_hot((a - b) // 10) + one_hot((a - b) % 10)]
                sequancesWithClasses.append(([right, left, minus, equals], target))
                target = [dontCare, dontCare, dontCare, one_hot((a - b) // 10) + one_hot((a - b) % 10)]
                sequancesWithoutClasses.append(([right, left, minus, equals], target))

            #divide
            if b > 0:
                target = [empty_one_hot + one_hot(b), empty_one_hot + one_hot(a), divide_vector, one_hot(a // b) + one_hot(a % b)]
                sequancesWithClasses.append(([right, left, divide, equals], target))
                target = [dontCare, dontCare, dontCare, one_hot(a // b) + one_hot(a % b)]
                sequancesWithoutClasses.append(([right, left, divide, equals], target))

np.random.shuffle(sequancesWithClasses)
np.random.shuffle(sequancesWithoutClasses)

split = int(len(sequancesWithClasses) * 0.2)

k = 5
k_fold_index = 0

scores = []

while k_fold_index < k:

    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    index = k_fold_index * split
    count = 0
    while count < len(sequancesWithoutClasses) - split:
        x, y = sequancesWithoutClasses[index]
        train_x.append(x)
        train_y.append(y)
        index = (index + 1) % len(sequancesWithoutClasses)
        count += 1
    count = 0
    while count < int(split / 2):
        x, y = sequancesWithoutClasses[index]
        val_x.append(x)
        val_y.append(y)
        index = (index + 1) % len(sequancesWithoutClasses)
        count += 1
    count = 0
    while count < int(split / 2):
        x, y = sequancesWithoutClasses[index]
        test_x.append(x)
        test_y.append(y)
        index = (index + 1) % len(sequancesWithoutClasses)
        count += 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    model = LSTMModel(4, [512, 512], 784, 20)
    model.train(train_x, train_y, val_x, val_y, 100, 100)

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

    image.save("results-without-classes-" + str(k_fold_index) + ".png")
    scores.append(count / float(total))

    k_fold_index += 1

print(scores)

#with-classes:
#[0.8782608695652174, 0.8347826086956521, 0.8869565217391304, 0.8608695652173913, 0.9072463768115943]

#without-classes:
#[0.7884057971014493, 0.808695652173913, 0.7681159420289855, 0.8405797101449275, 0.8318840579710145]