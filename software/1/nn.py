import tensorflow as tf
from utils import selectMinBatch


def constructNN(x, input_size, output_size, hidden_layers):
    weights = []
    biases = []
    layer_input_size = input_size
    previous_layer = x
    for hidden_layer_size in hidden_layers:
        W = tf.Variable(tf.truncated_normal([layer_input_size, hidden_layer_size]))
        b = tf.Variable(tf.zeros([hidden_layer_size]))
        weights.append(W)
        biases.append(b)
        hidden = tf.sigmoid(tf.matmul(previous_layer, W) + b)
        layer_input_size = hidden_layer_size
        previous_layer = hidden
    W_output = tf.Variable(tf.truncated_normal([layer_input_size, output_size]))
    b_output = tf.Variable(tf.zeros([output_size]))
    y = tf.sigmoid(tf.matmul(previous_layer, W_output) + b_output)
    weights.append(W_output)
    biases.append(b_output)
    return [weights, biases, y]

def trainNN(sess, loss, train_step, num_epochs, x, y, y_, weights, biases, trainData, trainTargets, validationData, validationTargets, use_min_batch = False):
    min_training_loss = 999999999
    min_validation_loss = 999999999
    opt_weights = []
    opt_biases = []
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        trainDict = {}
        if use_min_batch:
            [xs, ys] = selectMinBatch(trainData, trainTargets)
            trainDict = {x: xs, y_: ys}
        else:
            trainDict = {x: trainData, y_: trainTargets}
        sess.run(train_step, feed_dict = trainDict)
        if epoch % 10 == 0:
            [curr_loss, curr_weights, curr_biases] = sess.run([loss, weights, biases], feed_dict = trainDict)
            validation_loss = sess.run(loss, feed_dict = {x: validationData, y_: validationTargets})
            train_losses.append(curr_loss)
            validation_losses.append(validation_loss)
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                opt_weights = curr_weights
                opt_biases = curr_biases
            if curr_loss < min_training_loss:
                min_training_loss = curr_loss
            print("Epoch: " + str(epoch) + " - Trianing Loss: " + str(curr_loss) + " - Validation Loss: " + str(validation_loss))
    
    return [opt_weights, opt_biases, curr_weights, curr_biases, min_training_loss, min_validation_loss, train_losses, validation_losses]