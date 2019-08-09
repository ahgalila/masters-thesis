from __future__ import division, print_function, absolute_import

import tensorflow as tf
import sys, datetime

sys.dont_write_bytecode = True
import utils, data, nn

from PIL import Image, ImageDraw, ImageFont

# Load Datasets
[
    trainData,
    trainDataSymbols,
    trainTargets,
    trainTargetsSymbols,
    trainTargetClasses,
    testData,
    testDataSymbols,
    testTargets,
    testTargetsSymbols,
    validationData,
    validationDataSymbols,
    validationTargets,
    validationTargetsSymbols,
    validationTargetClasses 
] = data.loadDataSet()

testData = []
testTargets = []
for a in range(6, 10):
    for b in range(6, 10):
        testData.append(utils.one_hot(a) + utils.one_hot(b))
        testTargets.append((a + b) % 10)

# Construct Graph
x = tf.placeholder(tf.float32, [None, 20])
[weights, biases, y] = nn.constructNN(x, 20, 1568, [100, 100])
y_ = tf.placeholder(tf.float32, [None, 1568])

# Session Setup
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Train NN
num_epochs = 1000

[
    opt_weights,
    opt_biases,
    final_weights,
    final_biases,
    min_training_loss,
    min_validation_loss,
    train_losses,
    validation_losses
] = nn.trainNN(sess, loss, train_step, num_epochs, x, y, y_, weights, biases, trainDataSymbols, trainTargets, validationDataSymbols, validationTargets)

# Output - Optimum weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(opt_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(opt_biases[index]))
outputOpt = sess.run(y, feed_dict = {x: testData})

# Output - Final weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(final_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(final_biases[index]))
outputFinal = sess.run(y, feed_dict = {x: testData})

# Evaluate Model
[testScore, testResults] = utils.validate(outputOpt, testTargets)
[testScoreFinal, testResultsFinal] = utils.validate(outputFinal, testTargets)
print ("Score - Optimum: " + str(testScore))
print ("Score - Final: " + str(testScoreFinal))

image = Image.new('RGB', (56 * 21, 28 * 16))
draw = ImageDraw.Draw(image)
for row, output in enumerate(outputOpt):
    for y in range(28):
        for x in range(28):
            intensity = int(output[(y * 28 + x)] * 255)
            draw.point((x, y + row * 28), fill = (intensity, intensity, intensity))
    for y in range(28):
        for x in range(28):
            intensity = int(output[784:][(y * 28 + x)] * 255)
            draw.point((x + 28, y + row * 28), fill = (intensity, intensity, intensity))

    for trainTargetsIndex in range(20):
        for y in range(28):
            for x in range(28):
                intensity = int(trainTargets[(20 * row) + trainTargetsIndex][(y * 28 + x)] * 255)
                draw.point((x + 56 + (56 * trainTargetsIndex), y + row * 28), fill = (intensity, intensity, intensity))
        for y in range(28):
            for x in range(28):
                intensity = int(trainTargets[(20 * row) + trainTargetsIndex][784:][(y * 28 + x)] * 255)
                draw.point((x + 28 + 56 + (56 * trainTargetsIndex), y + row * 28), fill = (intensity, intensity, intensity))
image.save("outputs/symbols-noisy.png")
image.show()