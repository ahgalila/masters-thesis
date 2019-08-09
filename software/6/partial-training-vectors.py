from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from utils import one_hot, argmax, renderImage, renderDigit, getOperator

dontCare = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
plus = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

sequances = []

for a in range(10):
    for b in range(10):
        #target = [one_hot(0) + one_hot(b), one_hot(0) + one_hot(a), one_hot((a + b) // 10) + one_hot((a + b) % 10)]
        target = [dontCare, dontCare, one_hot((a + b) // 10) + one_hot((a + b) % 10)]
        sequances.append(([one_hot(a), one_hot(b), plus], target, a + b))

np.random.shuffle(sequances)

trainSequances = []
testSequances = []

for index, sequance in enumerate(sequances):
    if index % 4 == 0:
        testSequances.append(sequance)
    else:
        trainSequances.append(sequance)

train_x = []
train_y = []
for x, y, r in trainSequances:
    train_x.append(x)
    train_y.append(y)

test_x = []
test_y = []
test_result = []
for x, y, r in testSequances:
    test_x.append(x)
    test_y.append(y)
    test_result.append(r)

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

print(len(train_x))
print(len(test_x))

model = LSTMModel(3, [512, 512], 10, 20)
model.train(train_x, train_y, test_x, test_y, 10, 200)

results = model.predict(test_x)
count = 0
total = 0

for index, result in enumerate(results):
    left = argmax(result[2][:10])
    right = argmax(result[2][10:])

    if left * 10 + right == test_result[index]:
        count += 1
    total += 1

print("SCORE: " + str(count / float(total)))