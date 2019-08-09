from __future__ import division, print_function, absolute_import

import numpy as np
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

trainDataCombined = []
testDataCombined = []
validationDataCombined = []

for index, item in enumerate(trainData):
    trainDataCombined.append(np.concatenate((item, trainDataSymbols[index]), axis=0))
for index, item in enumerate(testData):
    testDataCombined.append(np.concatenate((item, testDataSymbols[index]), axis=0))
for index, item in enumerate(validationData):
    validationDataCombined.append(np.concatenate((item, validationDataSymbols[index]), axis=0))

# Construct Graph
x = tf.placeholder(tf.float32, [None, 1588])
[weights, biases, y] = nn.constructNN(x, 1588, 20, [100, 100])
y_ = tf.placeholder(tf.float32, [None, 20])

# Session Setup
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Train NN
num_epochs = 2000

[
    opt_weights,
    opt_biases,
    final_weights,
    final_biases,
    min_training_loss,
    min_validation_loss,
    train_losses,
    validation_losses
] = nn.trainNN(sess, loss, train_step, num_epochs, x, y, y_, weights, biases, trainDataCombined, trainTargetsSymbols, validationDataCombined, validationTargetsSymbols)

# Output - Optimum weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(opt_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(opt_biases[index]))
trainOutput = sess.run(y, feed_dict = {x: trainDataCombined})
validationOutput = sess.run(y, feed_dict = {x: validationDataCombined})
testOutput = sess.run(y, feed_dict = {x: testDataCombined})

# Output - Final weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(final_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(final_biases[index]))
trainOutputFinal = sess.run(y, feed_dict = {x: trainDataCombined})
validationOutputFinal = sess.run(y, feed_dict = {x: validationDataCombined})
testOutputFinal = sess.run(y, feed_dict = {x: testDataCombined})

# Evaluate Model
successCount = 0
for index, output in enumerate(trainOutput):
    if utils.argmax(output[10:]) == trainTargetClasses[index]:
        successCount = successCount + 1
print("Score - Train - Optimum: " + str(successCount / len(trainOutput)))

successCount = 0
for index, output in enumerate(validationOutput):
    if utils.argmax(output[10:]) == validationTargetClasses[index]:
        successCount = successCount + 1
print("Score - Validation - Optimum: " + str(successCount / len(validationOutput)))

successCount = 0
for index, output in enumerate(testOutput):
    if utils.argmax(output[10:]) == testTargets[index]:
        successCount = successCount + 1
print("Score - Test - Optimum: " + str(successCount / len(testOutput)))

successCount = 0
for index, output in enumerate(trainOutputFinal):
    if utils.argmax(output[10:]) == trainTargetClasses[index]:
        successCount = successCount + 1
print("Score - Train - Final: " + str(successCount / len(trainOutputFinal)))

successCount = 0
for index, output in enumerate(validationOutputFinal):
    if utils.argmax(output[10:]) == validationTargetClasses[index]:
        successCount = successCount + 1
print("Score - Validation - Final: " + str(successCount / len(validationOutputFinal)))

successCount = 0
for index, output in enumerate(testOutputFinal):
    if utils.argmax(output[10:]) == testTargets[index]:
        successCount = successCount + 1
print("Score - Test - Final: " + str(successCount / len(testOutputFinal)))