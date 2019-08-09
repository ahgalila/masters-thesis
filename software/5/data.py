from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

from tensorflow.examples.tutorials.mnist import input_data
from random import randint
from utils import argmax, one_hot
import numpy as np

class DataLoader(object):
    
    def __init__(self, src=None, sample_size=10):
        if src:
            self.__loadData(src)
        else:
            self.__generateData(sample_size)
    
    def __generateData(self, sample_size):
        
        self.data = {}
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trainSamples = [[], [], [], [], [], [], [], [], [], []]
        testSamples = [[], [], [], [], [], [], [], [], [], []]

        plus = np.load("MNIST_data/plus.npy")
        minus = np.load("MNIST_data/minus.npy")
        times = np.load("MNIST_data/times.npy")
        divide = np.load("MNIST_data/divide.npy")

        empty_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        plus_vector = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        times_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        minus_vector = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        divide_vector = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(len(mnist.train.images)):
            trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])
        for i in range(len(mnist.test.images)):
            testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])

        for a in range(0, 10):
            for b in range(0, 10):

                self.data[str(a) + "+" + str(b)] = []
                self.data[str(a) + "x" + str(b)] = []
                self.data[str(a) + "-" + str(b)] = []
                self.data[str(a) + "/" + str(b)] = []

                for i in range(sample_size):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    #plus
                    self.data[str(a) + "+" + str(b)].append({
                        "left": left,
                        "right": right,
                        "operator": plus,
                        "left-class": empty_one_hot + one_hot(a),
                        "right-class": empty_one_hot + one_hot(b),
                        "operator-class": plus_vector,
                        "target": one_hot((a + b) // 10) + one_hot((a + b) % 10),
                        "hasSymbol": False
                    })

                    #times
                    self.data[str(a) + "x" + str(b)].append({
                        "left": left,
                        "right": right,
                        "operator": times,
                        "left-class": empty_one_hot + one_hot(a),
                        "right-class": empty_one_hot + one_hot(b),
                        "operator-class": times_vector,
                        "target": one_hot((a * b) // 10) + one_hot((a * b) % 10)
                    })

                    #minus
                    if a >= b:
                        self.data[str(a) + "-" + str(b)].append({
                            "left": left,
                            "right": right,
                            "operator": minus,
                            "left-class": empty_one_hot + one_hot(a),
                            "right-class": empty_one_hot + one_hot(b),
                            "operator-class": minus_vector,
                            "target": one_hot((a - b) // 10) + one_hot((a - b) % 10)
                        })

                    #divide
                    if b > 0:
                        self.data[str(a) + "/" + str(b)].append({
                            "left": left,
                            "right": right,
                            "operator": divide,
                            "left-class": empty_one_hot + one_hot(a),
                            "right-class": empty_one_hot + one_hot(b),
                            "operator-class": divide_vector,
                            "target": one_hot(a // b) + one_hot(a % b)
                        })

    def __splitData(self, cross_validation_index, num_validation, num_test, num_with_symbols):
        temp = {}
        train = []
        validation = []
        test = []

        for operator in ["+", "x", "-", "/"]:
            for a in range(10):
                for b in range(10):
                    key = str(a) + operator + str(b)
                    temp[key] = []
                    for index, sample in enumerate(self.data[key]):
                        if index >= cross_validation_index * num_test and index < cross_validation_index * num_test + num_test:
                            test.append(sample)
                        else:
                            temp[key].append(sample)

        for key in temp:
            symbolsCount = 0
            for index, sample in enumerate(temp[key]):
                if index < num_validation:
                    validation.append(sample)
                else:
                    if symbolsCount < num_with_symbols:
                        sample["hasSymbol"] = True
                    else:
                        sample["hasSymbol"] = False
                    train.append(sample)
                    symbolsCount += 1

        np.random.shuffle(train)

        return train, validation, test

    def getData(self, cross_validation_index, num_validation, num_test, num_with_symbols):
        
        train, validation, test = self.__splitData(cross_validation_index, num_validation, num_test, num_with_symbols)
        
        equals = np.load("MNIST_data/equals.npy")
        dontCare = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        train_x = []
        train_y = []
        val_x = []
        val_y = []
        test_x = []
        test_y = []

        for sample in train:
            train_x.append([sample["right"], sample["left"], sample["operator"], equals])
            if sample["hasSymbol"]:
                train_y.append([sample["right-class"], sample["left-class"], sample["operator-class"], sample["target"]])
            else:
                train_y.append([dontCare, dontCare, dontCare, sample["target"]])

        for sample in validation:
            val_x.append([sample["right"], sample["left"], sample["operator"], equals])
            val_y.append([sample["right-class"], sample["left-class"], sample["operator-class"], sample["target"]])

        for sample in test:
            test_x.append([sample["right"], sample["left"], sample["operator"], equals])
            test_y.append([sample["right-class"], sample["left-class"], sample["operator-class"], sample["target"]])

        return [
            np.array(train_x),
            np.array(train_y),
            np.array(val_x),
            np.array(val_y),
            np.array(test_x),
            np.array(test_y)
        ]

    def __loadData(self, folder):
        
        data = np.load(folder + "/data.npy")
        self.data = data.item()

    def saveData(self, folder):
        
        np.save(folder + "/data", self.data)