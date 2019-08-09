from __future__ import division, print_function, absolute_import
import utils
import sys
sys.dont_write_bytecode = True

import numpy as np

from lstm_model import LSTMModel
from utils import argmax, one_hot, argmax, renderImage, getOperator
from PIL import Image, ImageDraw, ImageFont

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

plus = np.load("MNIST_data/plus.npy")
minus = np.load("MNIST_data/minus.npy")
times = np.load("MNIST_data/times.npy")
divide = np.load("MNIST_data/divide.npy")

empty_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dontCare = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
empty_operation = [0, 0, 0, 0]

sequances = []

for a in range(0, 10):
    for b in range(0, 10):
        
        leftSymbol = symbols[a]
        rightSymbol = symbols[b]

        #plus
        target = [dontCare, dontCare, [1, 0, 0, 0] + one_hot((a + b) // 10) + one_hot((a + b) % 10)]
        #target = [empty_operation + empty_one_hot + one_hot(b), empty_operation + empty_one_hot + one_hot(a), [1, 0, 0, 0] + one_hot((a + b) // 10) + one_hot((a + b) % 10)]
        sequances.append(([rightSymbol, leftSymbol, plus], target))

        #times
        target = [dontCare, dontCare, [0, 1, 0, 0] + one_hot((a * b) // 10) + one_hot((a * b) % 10)]
        #target = [empty_operation + empty_one_hot + one_hot(b), empty_operation + empty_one_hot + one_hot(a), [0, 1, 0, 0] + one_hot((a * b) // 10) + one_hot((a * b) % 10)]
        sequances.append(([rightSymbol, leftSymbol, times], target))

        #minus
        if a >= b:
            target = [dontCare, dontCare, [0, 0, 1, 0] + one_hot((a - b) // 10) + one_hot((a - b) % 10)]
            #target = [empty_operation + empty_one_hot + one_hot(b), empty_operation + empty_one_hot + one_hot(a), [0, 0, 1, 0] + one_hot((a - b) // 10) + one_hot((a - b) % 10)]
            sequances.append(([rightSymbol, leftSymbol, minus], target))

        #divide
        if b > 0:
            target = [dontCare, dontCare, [0, 0, 0, 1] + one_hot(a // b) + one_hot(a % b)]
            #target = [empty_operation + empty_one_hot + one_hot(b), empty_operation + empty_one_hot + one_hot(a), [0, 0, 0, 1] + one_hot(a // b) + one_hot(a % b)]
            sequances.append(([rightSymbol, leftSymbol, divide], target))

np.random.shuffle(sequances)

train_x = []
train_y = []
for x, y in sequances:
    train_x.append(x)
    train_y.append(y)

train_x = np.array(train_x)
train_y = np.array(train_y)

model = LSTMModel(3, [512], 784, 24)
model.load("checkpoint.hdf5")
#model.train(train_x, train_y, train_x, train_y, 10, 100)
#result = model.evaluate(train_x, train_y)

image = Image.new('RGB', (28 * 10, 28 * 100))
draw = ImageDraw.Draw(image)

count = 0
total = 0

index = 0
for a in range(10):
    for b in range(10):
        
        leftSymbol = symbols[a]
        rightSymbol = symbols[b]

        sequance = np.array([[rightSymbol, leftSymbol, times]])

        result = model.predict(sequance)

        left = argmax(result[0][2][-20:-10])
        right = argmax(result[0][2][-10:])

        if left * 10 + right == a * b:
            count += 1
        total += 1

        renderImage(image, "MNIST_data/" + str(a) + ".png", 0 * 28, index * 28)
        renderImage(image, "MNIST_data/times.png", 1 * 28, index * 28)
        renderImage(image, "MNIST_data/" + str(b) + ".png", 2 * 28, index * 28)
        renderImage(image, "MNIST_data/implies.png", 3 * 28, index * 28)
        renderImage(image, "MNIST_data/" + str(argmax(result[0][1][-10:])) + ".png", 4 * 28, index * 28)
        renderImage(image, "MNIST_data/" + getOperator(result[0][2][:4]) + ".png", 5 * 28, index * 28)
        renderImage(image, "MNIST_data/" + str(argmax(result[0][0][-10:])) + ".png", 6 * 28, index * 28)
        renderImage(image, "MNIST_data/equals.png", 7 * 28, index * 28)
        renderImage(image, "MNIST_data/" + str(argmax(result[0][2][-20:-10])) + ".png", 8 * 28, index * 28)
        renderImage(image, "MNIST_data/" + str(argmax(result[0][2][-10:])) + ".png", 9 * 28, index * 28)

        index += 1

image.save("results.png")
print(count / float(total))


#print(result)