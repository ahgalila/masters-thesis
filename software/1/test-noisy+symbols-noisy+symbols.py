from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import sys, datetime

sys.dont_write_bytecode = True
import utils, data, nn

def getDontCare(size = 1568):
    ret = []
    for i in range(size):
        ret.append(0.5)
    return ret

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
trainTargetsCombined = []
testDataCombined = []
validationDataCombined = []
validationTargetsCombined = []


for index, item in enumerate(trainData):
    trainDataCombined.append(np.concatenate((item, trainDataSymbols[index]), axis=0))
for i in range(70):
    for index, item in enumerate(trainData):
        trainDataCombined.append(np.concatenate((getDontCare(), trainDataSymbols[index]), axis=0))
for index, item in enumerate(trainTargets):
    trainTargetsCombined.append(np.concatenate((item, trainTargetsSymbols[index]), axis=0))
for i in range(70):
    for index, item in enumerate(trainTargets):
        trainTargetsCombined.append(np.concatenate((item, trainTargetsSymbols[index]), axis=0))
for index, item in enumerate(testData):
    testDataCombined.append(np.concatenate((item, testDataSymbols[index]), axis=0))
for index, item in enumerate(validationData):
    validationDataCombined.append(np.concatenate((item, validationDataSymbols[index]), axis=0))
for index, item in enumerate(validationTargets):
    validationTargetsCombined.append(np.concatenate((item, validationTargetsSymbols[index]), axis=0))


# Construct Graph
x = tf.placeholder(tf.float32, [None, 1588])
[weights, biases, y] = nn.constructNN(x, 1588, 1588, [100, 100, 100])
y_ = tf.placeholder(tf.float32, [None, 1588])

# Session Setup
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Train NN
num_epochs = 3000

[
    opt_weights,
    opt_biases,
    final_weights,
    final_biases,
    min_training_loss,
    min_validation_loss,
    train_losses,
    validation_losses
] = nn.trainNN(sess, loss, train_step, num_epochs, x, y, y_, weights, biases, trainDataCombined, trainTargetsCombined, validationDataCombined, validationTargetsCombined)

# Output - Final weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(final_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(final_biases[index]))
trainOutputFinal = sess.run(y, feed_dict = {x: trainDataCombined})
validationOutputFinal = sess.run(y, feed_dict = {x: validationDataCombined})
testOutputFinal = sess.run(y, feed_dict = {x: testDataCombined})

# Evaluate Model
name = "noisy+symbols-noisy_symbols - " + str(datetime.datetime.now())
utils.plot_performance(train_losses, validation_losses, num_epochs, "Noisy with Symbols to Noisy with Symbols", name)
#utils.render_output(validationOutputFinal, testOutputFinal, validationDataCombined, testDataCombined, validationTargetsCombined, name + "-final")
[testScoreFinal, testResultsFinal] = utils.validate(testOutputFinal, testTargets)
#[trainScoreFinal, trainResultsFinal] = utils.validate(trainOutputFinal, trainTargetClasses)
[validationScoreFinal, validationResultsFinal] = utils.validate(validationOutputFinal, validationTargetClasses)
testSymbolsScore = utils.validateSymbols(testOutputFinal, testTargetsSymbols)
f = open("summaries/" + name + ".txt", mode="w")
f.write("Noisy with Symbols to Noisy with Symbols - 2 hidden layers - 100 units each - 1000 epochs - AdamOptimizer - Learning Rate: 0.01 \n")
f.write("Minimum Training Loss: " + str(min_training_loss) + "\n")
f.write("Minimum Validation Loss: " + str(min_validation_loss) + "\n")
f.write("Test Score: " + str(testScoreFinal) + "\n")
f.write("Test Symbols Score: " + str(testSymbolsScore) + "\n")
index = 0
for result in testResultsFinal:
    f.write("Predicted: " + str(result) + " - Target: " + str(testTargets[index]) + "\n")
    index = index + 1
'''f.write("Train Score: " + str(trainScoreFinal) + "\n")
index = 0
for result in trainResultsFinal:
    f.write("Predicted: " + str(result) + " - Target: " + str(trainTargetClasses[index]) + "\n")
    index = index + 1'''
f.write("Validation Score: " + str(validationScoreFinal) + "\n")
index = 0
for result in validationResultsFinal:
    f.write("Predicted: " + str(result) + " - Target: " + str(validationTargetClasses[index]) + "\n")
    index = index + 1
f.close()