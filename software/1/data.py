from __future__ import division, print_function, absolute_import

import sys, math
sys.dont_write_bytecode = True

from tensorflow.examples.tutorials.mnist import input_data
from random import randint
from utils import argmax, one_hot, renderUnsupervisedData
import numpy as np

def loadDataSet():
    return [
        np.load("MNIST_data/train-data.npy"),
        np.load("MNIST_data/train-data-symbols.npy"),
        np.load("MNIST_data/train-targets.npy"),
        np.load("MNIST_data/train-targets-symbols.npy"),
        np.load("MNIST_data/train-target-classes.npy"),
        np.load("MNIST_data/test-data.npy"),
        np.load("MNIST_data/test-data-symbols.npy"),
        np.load("MNIST_data/test-targets.npy"),
        np.load("MNIST_data/test-targets-symbols.npy"),
        np.load("MNIST_data/validation-data.npy"),
        np.load("MNIST_data/validation-data-symbols.npy"),
        np.load("MNIST_data/validation-targets.npy"),
        np.load("MNIST_data/validation-targets-symbols.npy"),
        np.load("MNIST_data/validation-target-classes.npy")
    ]

def generateUnsupervisedDataSet(num = 10):
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trainSamples = [[], [], [], [], [], [], [], [], [], []]

    data = []

    for i in range(len(mnist.train.images)):
        trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])

    for digit in range(10):
        for i in range(num):
            sample = trainSamples[digit][randint(0, len(trainSamples[digit]) - 1)]
            item = []
            for pixel in sample:
                item.append(int(math.ceil(pixel)))
            data.append(item)
    
    np.save("MNIST_data/unsupervised-train-data", data)
    renderUnsupervisedData(data)

def generateDataSet(num_train = 10, num_test = 20, num_validation = 20):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trainSamples = [[], [], [], [], [], [], [], [], [], []]
    testSamples = [[], [], [], [], [], [], [], [], [], []]
    validationSamples = [[], [], [], [], [], [], [], [], [], []]

    trainData = []
    trainDataSymbols = []
    trainTargets = []
    trainTargetsSymbols = []
    trainTargetClasses = []
    testData = []
    testDataSymbols = []
    testTargets = []
    testTargetsSymbols = []
    validationData = []
    validationDataSymbols = []
    validationTargets = []
    validationTargetsSymbols = []
    validationTargetClasses = []

    for i in range(len(mnist.train.images)):
        trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])
    for i in range(len(mnist.test.images)):
        testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])
    for i in range(len(mnist.validation.images)):
        validationSamples[argmax(mnist.validation.labels[i])].append(mnist.validation.images[i])

    for a in range(0, 10):
        for b in range(0, 10):
            for i in range(num_train):
                left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]
                merge = []
                for i in range(len(left)):
                    merge.append(left[i])
                for i in range(len(right)):
                    merge.append(right[i])
                trainData.append(merge)
                trainDataSymbols.append(one_hot(a) + one_hot(b))
                left = trainSamples[(a + b) // 10][randint(0, len(trainSamples[(a + b) // 10]) - 1)]
                right = trainSamples[(a + b) % 10][randint(0, len(trainSamples[(a + b) % 10]) - 1)]
                merge = []
                for i in range(len(left)):
                    merge.append(left[i])
                for i in range(len(right)):
                    merge.append(right[i])
                trainTargets.append(merge)
                trainTargetsSymbols.append(one_hot((a + b) // 10) + one_hot((a + b) % 10))
                trainTargetClasses.append((a + b) % 10)
            for i in range(num_validation):
                left = validationSamples[a][randint(0, len(validationSamples[a]) - 1)]
                right = validationSamples[b][randint(0, len(validationSamples[b]) - 1)]
                merge = []
                for i in range(len(left)):
                    merge.append(left[i])
                for i in range(len(right)):
                    merge.append(right[i])
                validationData.append(merge)
                validationDataSymbols.append(one_hot(a) + one_hot(b))
                left = validationSamples[(a + b) // 10][randint(0, len(validationSamples[(a + b) // 10]) - 1)]
                right = validationSamples[(a + b) % 10][randint(0, len(validationSamples[(a + b) % 10]) - 1)]
                merge = []
                for i in range(len(left)):
                    merge.append(left[i])
                for i in range(len(right)):
                    merge.append(right[i])
                validationTargets.append(merge)
                validationTargetsSymbols.append(one_hot((a + b) // 10) + one_hot((a + b) % 10))
                validationTargetClasses.append((a + b) % 10)
            for i in range(num_test):
                left = testSamples[a][randint(0, len(testSamples[a]) - 1)]
                right = testSamples[b][randint(0, len(testSamples[b]) - 1)]
                merge = []
                for i in range(len(left)):
                    merge.append(left[i])
                for i in range(len(right)):
                    merge.append(right[i])
                testData.append(merge)
                testDataSymbols.append(one_hot(a) + one_hot(b))
                testTargets.append((a + b) % 10)
                testTargetsSymbols.append(one_hot((a + b) // 10) + one_hot((a + b) % 10))

    np.save("MNIST_data/train-data", trainData)
    np.save("MNIST_data/train-data-symbols", trainDataSymbols)
    np.save("MNIST_data/train-targets", trainTargets)
    np.save("MNIST_data/train-targets-symbols", trainTargetsSymbols)
    np.save("MNIST_data/train-target-classes", trainTargetClasses)
    np.save("MNIST_data/test-data", testData)
    np.save("MNIST_data/test-data-symbols", testDataSymbols)
    np.save("MNIST_data/test-targets", testTargets)
    np.save("MNIST_data/test-targets-symbols", testTargetsSymbols)
    np.save("MNIST_data/validation-data", validationData)
    np.save("MNIST_data/validation-data-symbols", validationDataSymbols)
    np.save("MNIST_data/validation-targets", validationTargets)
    np.save("MNIST_data/validation-targets-symbols", validationTargetsSymbols)
    np.save("MNIST_data/validation-target-classes", validationTargetClasses)

def generateLSTMDataSet(num_train = 50, num_test = 20, num_symbols = 50):
    
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trainSamples = [[], [], [], [], [], [], [], [], [], []]
    testSamples = [[], [], [], [], [], [], [], [], [], []]

    for i in range(len(mnist.train.images)):
        trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])
    for i in range(len(mnist.test.images)):
        testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])

    trainNoisy = []
    trainNoisySymbols = []
    trainTargets = []

    testNoisy = []
    testNoisySymbols = []
    testTargets = []

    symbolsIn = []
    symbolsOut = []

    for a in range(0, 10):
        for b in range(0, 10):
            
            leftSymbol = one_hot(a)
            rightSymbol = one_hot(b)

            target = one_hot((a + b) // 10) + one_hot((a + b) % 10)

            symbolsIn.append([leftSymbol, rightSymbol])
            symbolsOut.append(target)

            for i in range(num_train):
                left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]
                trainNoisy.append([left, right])

                trainNoisySymbols.append([np.concatenate((left, leftSymbol)), np.concatenate((right, rightSymbol))])

                trainTargets.append(target)

            for i in range(num_test):
                left = testSamples[a][randint(0, len(testSamples[a]) - 1)]
                right = testSamples[b][randint(0, len(testSamples[b]) - 1)]
                testNoisy.append([left, right])

                testNoisySymbols.append([np.concatenate((left, leftSymbol)), np.concatenate((right, rightSymbol))])

                testTargets.append(target)

    np.save("MNIST_data/lstm/train-noisy", trainNoisy)
    np.save("MNIST_data/lstm/train-noisy-symbols", trainNoisySymbols)
    np.save("MNIST_data/lstm/train-targets", trainTargets)

    np.save("MNIST_data/lstm/test-noisy", testNoisy)
    np.save("MNIST_data/lstm/test-noisy-symbols", testNoisySymbols)
    np.save("MNIST_data/lstm/test-targets", testTargets)

    np.save("MNIST_data/lstm/symbols-in", symbolsIn)
    np.save("MNIST_data/lstm/symbols-out", symbolsOut)

def loadLSTMDataSet():
    return [
        np.load("MNIST_data/lstm/train-noisy")
        np.load("MNIST_data/lstm/train-noisy-symbols")
        np.load("MNIST_data/lstm/train-targets")

        np.load("MNIST_data/lstm/test-noisy")
        np.load("MNIST_data/lstm/test-noisy-symbols")
        np.load("MNIST_data/lstm/test-targets")

        np.load("MNIST_data/lstm/symbols-in")
        np.load("MNIST_data/lstm/symbols-out")
    ]

#generateLSTMDataSet()

#generateDataSet()
#generateUnsupervisedDataSet()