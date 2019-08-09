from __future__ import division, print_function, absolute_import

import tensorflow as tf
import sys, datetime

sys.dont_write_bytecode = True
import utils, data, nn

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


# Construct Graph
x = tf.placeholder(tf.float32, [None, 1568])
[weights, biases, y] = nn.constructNN(x, 1568, 1568, [100, 100])
y_ = tf.placeholder(tf.float32, [None, 1568])

# Session Setup
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Train NN
num_epochs = 5000

[
    opt_weights,
    opt_biases,
    final_weights,
    final_biases,
    min_training_loss,
    min_validation_loss,
    train_losses,
    validation_losses
] = nn.trainNN(sess, loss, train_step, num_epochs, x, y, y_, weights, biases, trainData, trainTargets, validationData, validationTargets)

# Output - Optimum weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(opt_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(opt_biases[index]))
trainOutput = sess.run(y, feed_dict = {x: trainData})
validationOutput = sess.run(y, feed_dict = {x: validationData})
testOutput = sess.run(y, feed_dict = {x: testData})

# Output - Final weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(final_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(final_biases[index]))
trainOutputFinal = sess.run(y, feed_dict = {x: trainData})
validationOutputFinal = sess.run(y, feed_dict = {x: validationData})
testOutputFinal = sess.run(y, feed_dict = {x: testData})

# Evaluate Model
name = "noisy-noisy - " + str(datetime.datetime.now())
utils.plot_performance(train_losses, validation_losses, num_epochs, "Noisy to Noisy", name)
utils.render_output(validationOutput, testOutput, validationData, testData, validationTargets, name + "-optimum")
utils.render_output(validationOutputFinal, testOutputFinal, validationData, testData, validationTargets, name + "-final")
[testScore, testResults] = utils.validate(testOutput, testTargets)
[testScoreFinal, testResultsFinal] = utils.validate(testOutputFinal, testTargets)
[trainScore, trainResults] = utils.validate(trainOutput, trainTargetClasses)
[trainScoreFinal, trainResultsFinal] = utils.validate(trainOutputFinal, trainTargetClasses)
[validationScore, validationResults] = utils.validate(validationOutput, validationTargetClasses)
[validationScoreFinal, validationResultsFinal] = utils.validate(validationOutputFinal, validationTargetClasses)
f = open("summaries/" + name + ".txt", mode="w")
f.write("Noisy to Noisy - 2 hidden layers - 100 units each - 5000 epochs - AdamOptimizer - 0.001 Learing Rate - 1000 Training Examples per Combination \n")
f.write("Minimum Training Loss: " + str(min_training_loss) + "\n")
f.write("Minimum Validation Loss: " + str(min_validation_loss) + "\n")
f.write("Test Score - Optimum: " + str(testScore) + "\n")
f.write("Test Score - Final: " + str(testScoreFinal) + "\n")
index = 0
for result in testResults:
    f.write("Predicted: " + str(result) + " - Target: " + str(testTargets[index]) + "\n")
    index = index + 1
f.write("Train Score - Optimum: " + str(trainScore) + "\n")
f.write("Train Score - Final: " + str(trainScoreFinal) + "\n")
index = 0
for result in trainResults:
    f.write("Predicted: " + str(result) + " - Target: " + str(trainTargetClasses[index]) + "\n")
    index = index + 1
f.write("Validation Score - Optimum: " + str(validationScore) + "\n")
f.write("Validation Score - Final: " + str(validationScoreFinal) + "\n")
index = 0
for result in validationResults:
    f.write("Predicted: " + str(result) + " - Target: " + str(validationTargetClasses[index]) + "\n")
    index = index + 1
f.close()