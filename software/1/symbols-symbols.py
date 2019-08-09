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
x = tf.placeholder(tf.float32, [None, 20])
[weights, biases, y] = nn.constructNN(x, 20, 20, [256, 256])
y_ = tf.placeholder(tf.float32, [None, 20])

# Session Setup
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

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
] = nn.trainNN(sess, loss, train_step, num_epochs, y, y_, weights, biases, {x: trainDataSymbols, y_: trainTargetsSymbols}, {x: validationDataSymbols, y_: validationTargetsSymbols})

# Output - Optimum weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(opt_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(opt_biases[index]))
trainOutput = sess.run(y, feed_dict = {x: trainDataSymbols})
validationOutput = sess.run(y, feed_dict = {x: validationDataSymbols})
testOutput = sess.run(y, feed_dict = {x: testDataSymbols})

# Output - Final weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(final_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(final_biases[index]))
trainOutputFinal = sess.run(y, feed_dict = {x: trainDataSymbols})
validationOutputFinal = sess.run(y, feed_dict = {x: validationDataSymbols})
testOutputFinal = sess.run(y, feed_dict = {x: testDataSymbols})

# Evaluate Model
name = "symbols-symbols - " + str(datetime.datetime.now())
utils.plot_performance(train_losses, validation_losses, num_epochs, "Symbols to Symbols", name)
testScore = utils.validateSymbols(testOutput, testTargetsSymbols)
testScoreFinal = utils.validateSymbols(testOutputFinal, testTargetsSymbols)
trainScore = utils.validateSymbols(trainOutput, trainTargetsSymbols)
trainScoreFinal = utils.validateSymbols(trainOutputFinal, trainTargetsSymbols)
validationScore = utils.validateSymbols(validationOutput, validationTargetsSymbols)
validationScoreFinal = utils.validateSymbols(validationOutputFinal, validationTargetsSymbols)
f = open("summaries/" + name + ".txt", mode="w")
f.write("Symbols to Symbols - 2 hidden layers - 256 units each - 5000 epochs - GradientDescentOptimizer \n")
f.write("Minimum Training Loss: " + str(min_training_loss) + "\n")
f.write("Minimum Validation Loss: " + str(min_validation_loss) + "\n")
f.write("Test Score - Optimum: " + str(testScore) + "\n")
f.write("Test Score - Final: " + str(testScoreFinal) + "\n")
f.write("Train Score - Optimum: " + str(trainScore) + "\n")
f.write("Train Score - Final: " + str(trainScoreFinal) + "\n")
f.write("Validation Score - Optimum: " + str(validationScore) + "\n")
f.write("Validation Score - Final: " + str(validationScoreFinal) + "\n")
f.close()