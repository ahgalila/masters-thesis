from __future__ import division, print_function, absolute_import
from lstm_model import LSTMModel
from tensorflow.examples.tutorials.mnist import input_data
from utils import argmax, one_hot
from random import randint
import numpy as np

def temperature(value):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = -1
    while index >= -value:
        result[index] = 1
        index -= 1
    return result


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trainSamples = [[], [], [], [], [], [], [], [], [], []]
operators = {
    "+": np.load("MNIST_data/plus.npy"),
    "x": np.load("MNIST_data/times.npy"),
    "-": np.load("MNIST_data/minus.npy"),
    "/": np.load("MNIST_data/divide.npy")
}
plus = np.load("MNIST_data/plus.npy")
equals = np.load("MNIST_data/equals.npy")

for i in range(len(mnist.train.images)):
    trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])

train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []

testWatchListA = {}
testWatchListB = {}

for a in range(10):
    testWatchListA[str(a)] = randint(0, 9)
for b in range(10):
    testWatchListB[str(b)] = randint(0, 9)

for a in range(0, 10):
    for b in range(0, 10):
        for key in operators:
            if key == "+":
                leftTarget = [((a + b) // 10) / 10.0]
                rightTarget = [((a + b) % 10) / 10.0]
            if key == "x":
                leftTarget = [((a * b) // 10) / 10.0]
                rightTarget = [((a * b) % 10) / 10.0]
            if key == "-" and a >= b:
                leftTarget = [((a - b) // 10) / 10.0]
                rightTarget = [((a - b) % 10) / 10.0]
            if key == "/" and b > 0:
                leftTarget = [(a % b) / 10.0]
                rightTarget = [(a // b) / 10.0]
            
            if b == testWatchListA[str(a)] or a == testWatchListB[str(b)]:
                for i in range(2):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    test_x.append([right, left, operators[key], equals])
                    test_y.append([[b / 10.0], [a / 10.0], rightTarget, leftTarget])

            else:
                for i in range(25):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    train_x.append([right, left, operators[key], equals])
                    train_y.append([[b / 10.0], [a / 10.0], rightTarget, leftTarget])

                for i in range(2):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    val_x.append([right, left, operators[key], equals])
                    val_y.append([[b / 10.0], [a / 10.0], rightTarget, leftTarget])

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [512, 512], 784, 1)
model.train(train_x, train_y, val_x, val_y, 100, 5000)

results = model.predict(val_x)

count = 0

for index, result in enumerate(results):
    left = round(result[2][0], 1)
    right = round(result[3][0], 1)
    leftTarget = round(val_y[index][2][0], 1)
    rightTarget = round(val_y[index][3][0], 1)

    if left == leftTarget and right == rightTarget:
        count += 1

test_count = 0

test_results = model.predict(test_x)
for index, result in enumerate(test_results):
    left = round(result[2][0], 1)
    right = round(result[3][0], 1)
    leftTarget = round(test_y[index][2][0], 1)
    rightTarget = round(test_y[index][3][0], 1)

    if left == leftTarget and right == rightTarget:
        test_count += 1

print("SCORE: " + str(float(count) / len(results)))
print("TEST SCORE: " + str(float(test_count) / len(test_results)))

#SCORE: 0.882716049383
#TEST SCORE: 0.842105263158

# 100 per combination - 100%
#SCORE: 0.8875
#TEST SCORE: 0.675

# 20 per combination - 50%
#SCORE: 0.037037037037
#TEST SCORE: 0.0789473684211

# 20 per combination - 100%
#SCORE: 0.237804878049
#TEST SCORE: 0.111111111111