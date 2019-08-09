from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data

from flask import Flask, request, Response
import json

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 784])
hidden1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(hidden1, 0.8)
hidden2 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                     regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(hidden2, 0.8)
softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                             loss='categorical_crossentropy')

# Training
validator = tflearn.DNN(net, tensorboard_verbose=0)
validator.fit(mnist.train.images, mnist.train.labels, n_epoch=20, run_id="validator_model")

score = validator.evaluate(mnist.test.images, mnist.test.labels)
print('Validator Score: {}'.format(score))

app = Flask(__name__)
@app.route("/api/predict", methods=['POST'])
def predict():
    data = request.data.decode("utf-8")
    params = json.loads(data)
    json_ret =  json.dumps({'y': validator.predict(params['x']).tolist()})
    return Response(json_ret, mimetype = 'application/json')

app.run()
