from __future__ import division, print_function, absolute_import

import tensorflow as tf
import sys, datetime

sys.dont_write_bytecode = True
import utils, data, nn

# Load Datasets
trainData = []
trainTargets = []
for a in range(6, 10):
    for b in range(6, 10):
        trainData.append(utils.one_hot(a) + utils.one_hot(b))
        trainTargets.append(utils.one_hot((a + b) // 10) + utils.one_hot((a + b) % 10))

# Construct Graph
x = tf.placeholder(tf.float32, [None, 20])
[weights, biases, y] = nn.constructNN(x, 20, 20, [100, 100])
y_ = tf.placeholder(tf.float32, [None, 20])

# Session Setup
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ - y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Train NN
num_epochs = 100000

[
    opt_weights,
    opt_biases,
    final_weights,
    final_biases,
    min_training_loss,
    min_validation_loss,
    train_losses,
    validation_losses
] = nn.trainNN(sess, loss, train_step, num_epochs, x, y, y_, weights, biases, trainData, trainTargets, trainData, trainTargets)

# Output - Optimum weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(opt_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(opt_biases[index]))
outputOpt = sess.run(y, feed_dict = {x: trainData})

# Output - Final weights
for index, weight in enumerate(weights):
    sess.run(weight.assign(final_weights[index]))
for index, bias in enumerate(biases):
    sess.run(bias.assign(final_biases[index]))
outputFinal = sess.run(y, feed_dict = {x: trainData})

# Evaluate Model
successCount = 0
index = 0
for output in outputOpt:
    if utils.argmax(output[10:]) == utils.argmax(trainTargets[index][10:]):
        successCount = successCount + 1
    index = index + 1
score = successCount / len(outputOpt)
print ("Score: " + str(score))