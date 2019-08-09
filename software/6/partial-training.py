from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from utils import one_hot, argmax, renderImage, renderDigit, getOperator
from PIL import Image, ImageDraw, ImageFont

plus = np.load("MNIST_data/plus.npy")
equals = np.load("MNIST_data/equals.npy")
symbols = [
    np.load("MNIST_data/0.npy"),
    np.load("MNIST_data/1.npy"),
    np.load("MNIST_data/2.npy"),
    np.load("MNIST_data/3.npy"),
    np.load("MNIST_data/4.npy"),
    np.load("MNIST_data/5.npy"),
    np.load("MNIST_data/6.npy"),
    np.load("MNIST_data/7.npy"),
    np.load("MNIST_data/8.npy"),
    np.load("MNIST_data/9.npy")
]

dontCare = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
plus_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

sequances = []

for a in range(10):

    for b in range(10):
            
        leftSymbol = symbols[a]
        rightSymbol = symbols[b]

        target = [one_hot(0) + one_hot(b), one_hot(0) + one_hot(a), plus_vector, one_hot((a + b) // 10) + one_hot((a + b) % 10)]
        sequances.append(([rightSymbol, leftSymbol, plus, equals], target, a + b))

np.random.shuffle(sequances)

train_x = []
train_y = []
for x, y, r in sequances[:50]:
    train_x.append(x)
    train_y.append(y)

val_x = []
val_y = []
for x, y, r in sequances[50:75]:
    val_x.append(x)
    val_y.append(y)

test_x = []
test_y = []
for x, y, r in sequances[75:]:
    test_x.append(x)
    test_y.append(r)

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)
test_x = np.array(test_x)

print(len(train_x))
print(len(test_x))

model = LSTMModel(4, [512, 512, 512, 512], 784, 20)
model.train(train_x, train_y, val_x, val_y, 10, 200)

results = model.predict(test_x)
count = 0
total = 0

image = Image.new('RGB', (28 * 10, 28 * len(results)))
draw = ImageDraw.Draw(image)

for index, result in enumerate(results):
    left = argmax(result[3][:10])
    right = argmax(result[3][10:])

    if left * 10 + right == test_y[index]:
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

print("SCORE: " + str(count / float(total)))
image.save("results-partial.png")