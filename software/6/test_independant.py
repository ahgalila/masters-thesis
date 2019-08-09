from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

from tensorflow.examples.tutorials.mnist import input_data
from random import randint
from utils import argmax, one_hot
import numpy as np

class TestDataLoader(object):

    def __init__(self, src=None, size=5):
        if src:
            self.__loadData(src)
        else:
            self.__generateData(size)
    
    def __generateData(self, size):

        self.data = {}

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        testSamples = [[], [], [], [], [], [], [], [], [], []]

        plus = np.load("MNIST_data/plus.npy")
        minus = np.load("MNIST_data/minus.npy")
        times = np.load("MNIST_data/times.npy")
        divide = np.load("MNIST_data/divide.npy")

        plus_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        times_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        minus_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        divide_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        for i in range(len(mnist.test.images)):
            testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])

        for a in range(0, 10):
            for b in range(0, 10):

                self.data[str(a) + "+" + str(b)] = []
                self.data[str(a) + "x" + str(b)] = []
                self.data[str(a) + "-" + str(b)] = []
                self.data[str(a) + "/" + str(b)] = []

                for i in range(size):

                    left = testSamples[a][randint(0, len(testSamples[a]) - 1)]
                    right = testSamples[b][randint(0, len(testSamples[b]) - 1)]

                    #plus
                    self.data[str(a) + "+" + str(b)].append({
                        "left": left,
                        "right": right,
                        "operator": plus,
                        "left-class": one_hot(a),
                        "right-class": one_hot(b),
                        "operator-class": plus_vector,
                        "target": one_hot((a + b) // 10) + one_hot((a + b) % 10),
                        "hasSymbol": False
                    })

                    #times
                    self.data[str(a) + "x" + str(b)].append({
                        "left": left,
                        "right": right,
                        "operator": times,
                        "left-class": one_hot(a),
                        "right-class": one_hot(b),
                        "operator-class": times_vector,
                        "target": one_hot((a * b) // 10) + one_hot((a * b) % 10)
                    })

                    #minus
                    if a >= b:
                        self.data[str(a) + "-" + str(b)].append({
                            "left": left,
                            "right": right,
                            "operator": minus,
                            "left-class": one_hot(a),
                            "right-class": one_hot(b),
                            "operator-class": minus_vector,
                            "target": one_hot((a - b) // 10) + one_hot((a - b) % 10)
                        })

                    #divide
                    if b > 0:
                        self.data[str(a) + "/" + str(b)].append({
                            "left": left,
                            "right": right,
                            "operator": divide,
                            "left-class": one_hot(a),
                            "right-class": one_hot(b),
                            "operator-class": divide_vector,
                            "target": one_hot(a // b) + one_hot(a % b)
                        })

    def getData(self, operator, startA, endA, startB, endB):

        test_x = []
        test_y = []

        equals = np.load("MNIST_data/equals.npy")

        for a in range(startA, endA):
            for b in range(startB, endB):
                key = str(a) + operator + str(b)
                for sample in self.data[key]:
                    test_x.append([sample["right"], sample["left"], sample["operator"], equals])
                    test_y.append([sample["right-class"], sample["left-class"], sample["operator-class"], sample["target"]])

        return [
            np.array(test_x),
            np.array(test_y)
        ]

    def __loadData(self, folder):
        
        data = np.load(folder + "/test.npy")
        self.data = data.item()

    def saveData(self, folder):
        
        np.save(folder + "/test", self.data)