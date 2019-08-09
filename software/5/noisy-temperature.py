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

def is_temperature(value, target):
    for i in range(len(value)):
        if (target[1] == 1 and value[1] < 0.5) or target[i] == 0 and value[i] > 0.5:
            return False
    return True


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
                leftTarget = temperature((a + b) // 10)
                rightTarget = temperature((a + b) % 10)
            if key == "x":
                leftTarget = temperature((a * b) // 10)
                rightTarget = temperature((a * b) % 10)
            if key == "-" and a >= b:
                leftTarget = temperature((a - b) // 10)
                rightTarget = temperature((a - b) % 10)
            if key == "/" and b > 0:
                leftTarget = temperature(a % b)
                rightTarget = temperature(a // b)
            
            if b == testWatchListA[str(a)] or a == testWatchListB[str(b)]:
                for i in range(2):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    test_x.append([right, left, operators[key], equals])
                    test_y.append([temperature(b), temperature(a), rightTarget, leftTarget])

            else:
                for i in range(25):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    train_x.append([right, left, operators[key], equals])
                    train_y.append([temperature(b), temperature(a), rightTarget, leftTarget])

                for i in range(2):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    val_x.append([right, left, operators[key], equals])
                    val_y.append([temperature(b), temperature(a), rightTarget, leftTarget])

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [10, 100], 784, 9)
model.train(train_x, train_y, val_x, val_y, 100, 5000)

results = model.predict(val_x)

count = 0

for index, result in enumerate(results):
    left = result[2]
    right = result[3]
    leftTarget = val_y[index][2]
    rightTarget = val_y[index][3]

    if is_temperature(left, leftTarget) and is_temperature(right, rightTarget):
        count += 1

test_count = 0

test_results = model.predict(test_x)
for index, result in enumerate(test_results):
    left = result[2]
    right = result[3]
    leftTarget = test_y[index][2]
    rightTarget = test_y[index][3]

    if is_temperature(left, leftTarget) and is_temperature(right, rightTarget):
        test_count += 1

print("SCORE: " + str(float(count) / len(results)))
print("TEST SCORE: " + str(float(test_count) / len(test_results)))

#2 Layers, 512 units each
#SCORE: 0.716049382716
#TEST SCORE: 0.335526315789

#2 Layers, 20 units each
#SCORE: 0.54475308642
#TEST SCORE: 0.447368421053

#2 Layers, 100 units, 10 units
#SCORE: 0.7125
#TEST SCORE: 0.5375