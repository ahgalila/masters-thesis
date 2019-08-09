from __future__ import division, print_function, absolute_import
from lstm_model import LSTMModel
import numpy as np
from random import randint
import sys

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
            test_x.append([temperature(b) + [1, 0, 0], temperature(a) + [1, 0, 0], temperature(0) + [0, 1, 0], temperature(0) + [0, 0, 1]])
            test_y.append([temperature(b), temperature(a), temperature((a + b) % 10), temperature((a + b) // 10)])
        else:
            x.append([temperature(b) + [1, 0, 0], temperature(a) + [1, 0, 0], temperature(0) + [0, 1, 0], temperature(0) + [0, 0, 1]])
            y.append([temperature(b), temperature(a), temperature((a + b) % 10), temperature((a + b) // 10)])

x = np.array(x)
y = np.array(y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [5, 5], 12, 9)
model.train(x, y, test_x, test_y, 10, 5000)

results = model.predict(x)

count = 0

for index, result in enumerate(results):
    left = result[2]
    right = result[3]
    leftTarget = y[index][2]
    rightTarget = y[index][3]

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

#2 Hidden Layers, 512 Each
#SCORE: 0.95
#TEST SCORE: 0.6

#3 Hidden Layers, 512 Each
#SCORE: 0.95
#TEST SCORE: 0.75

#2 Hidden Layers, 20 Each
#SCORE: 1.0
#TEST SCORE: 0.842105263158

#2 Hidden Layers, 10 Each
#SCORE: 1.0
#TEST SCORE: 0.9