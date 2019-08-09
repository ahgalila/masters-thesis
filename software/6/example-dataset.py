
from tensorflow.examples.tutorials.mnist import input_data
from utils import argmax, renderImage, renderDigit, getOperator
from PIL import Image, ImageDraw, ImageFont
from random import randint
import numpy as np

image = Image.new('RGB', (28 * 3, 28 * 4))
draw = ImageDraw.Draw(image)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
samples = [[], [], [], [], [], [], [], [], [], []]

for i in range(len(mnist.train.images)):
    samples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])

plus = np.load("MNIST_data/plus.npy")

index = 0
for a in range(8, 10):
    for b in range(8, 10):
        left = samples[a][randint(0, len(samples[a]) - 1)]
        right = samples[b][randint(0, len(samples[b]) - 1)]
        renderDigit(draw, left, 0 * 28, index * 28)
        renderDigit(draw, plus, 1 * 28, index * 28)
        renderDigit(draw, right, 2 * 28, index * 28)

        index += 1

image.save("dataset-examples-5.png")