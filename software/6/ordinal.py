from __future__ import division, print_function, absolute_import
from lstm_model import LSTMModel
import numpy as np
from random import randint

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
            test_x.append([[b / 10.0, 0.0], [a / 10.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
            test_y.append([[b / 10.0], [a / 10.0], [((a + b) % 10) / 10.0], [((a + b) // 10) / 10.0]])
        else:
            x.append([[b / 10.0, 0.0], [a / 10.0, 0.0], [0.0, 0.5], [0.0, 1.0]])
            y.append([[b / 10.0], [a / 10.0], [((a + b) % 10) / 10.0], [((a + b) // 10) / 10.0]])

x = np.array(x)
y = np.array(y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [10, 10], 2, 1)
model.train(x, y, test_x, test_y, 10, 5000)

results = model.predict(x)

count = 0

for index, result in enumerate(results):
    left = round(result[2][0], 1)
    right = round(result[3][0], 1)
    leftTarget = round(y[index][2][0], 1)
    rightTarget = round(y[index][3][0], 1)

    if left == leftTarget and right == rightTarget:
        count += 1

test_count = 0

test_results = model.predict(test_x)
with open("ordinal-unseen-2.txt", "w+") as f:
    for index, result in enumerate(test_results):
        left = round(result[2][0], 1)
        right = round(result[3][0], 1)
        leftTarget = round(test_y[index][2][0], 1)
        rightTarget = round(test_y[index][3][0], 1)

        if left == leftTarget and right == rightTarget:
            test_count += 1

        f.write(str(test_x[index][0][0]) + " => " + str(result[0][0]) + "\n")
        f.write(str(test_x[index][1][0]) + " => " + str(result[1][0]) + "\n")
        f.write("+" + " => " + str(result[2][0]) + "\n")
        f.write("=" + " => " + str(result[3][0]) + "\n")
        f.write("\n\n")

print("SCORE: " + str(float(count) / len(results)))
print("TEST SCORE: " + str(float(test_count) / len(test_results)))