from __future__ import division, print_function, absolute_import

import random
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

        symbolsIn = []
        symbolsOut = []
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trainSamples = [[], [], [], [], [], [], [], [], [], []]
        testSamples = [[], [], [], [], [], [], [], [], [], []]

        symbols = [
            np.load("MNIST_data/0.npy"),
            np.load("MNIST_data/1.npy"),
            np.load("MNIST_data/2.npy"),
            np.load("MNIST_data/3.npy"),
            np.load("MNIST_data/4.npy"),
            np.load("MNIST_data/5.npy"),
            np.load("MNIST_data/6.npy"),
            np.load("MNIST_data/7.npy"),
            np.load("MNIST_data/8.npy"),
            np.load("MNIST_data/9.npy")
        ]

        for i in range(len(mnist.train.images)):
            trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])
        for i in range(len(mnist.test.images)):
            testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])

        for a in range(0, 10):
            for b in range(0, 10):
                
                self.data[str(a) + "+" + str(b)] = []
            
                leftSymbol = symbols[a]
                rightSymbol = symbols[b]

                target = one_hot((a + b) // 10) + one_hot((a + b) % 10)

                symbolsIn.append([leftSymbol, rightSymbol])
                symbolsOut.append(target)

                for i in range(sample_size):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    self.data[str(a) + "+" + str(b)].append({
                        "left-noisy": left,
                        "left-symbol": leftSymbol,
                        "right-noisy": right,
                        "right-symbol": rightSymbol,
                        "target": target
                    })

        self.symbolsIn = np.array(symbolsIn)
        self.symbolsOut = np.array(symbolsOut)

    def __splitData(self, cross_validation_index, num_validation, num_test):
        temp = {}
        train = []
        validation = []
        test = []

        for a in range(10):
            for b in range(10):
                key = str(a) + "+" + str(b)
                temp[key] = []
                for index, sample in enumerate(self.data[key]):
                    if index >= cross_validation_index * num_test and index < cross_validation_index * num_test + num_test:
                        test.append(sample)
                    else:
                        temp[key].append(sample)

        for key in temp:
            for index, sample in enumerate(temp[key]):
                if index < num_validation:
                    validation.append(sample)
                else:
                    train.append(sample)

        np.random.shuffle(train)
        #np.random.shuffle(validation)
        #np.random.shuffle(test)

        return train, validation, test

    def getSequencialData(self, cross_validation_index, num_validation, num_test):
        
        train, validation, test = self.__splitData(cross_validation_index, num_validation, num_test)
        
        noisyTrain = []
        noisySymbolsTrain = []
        targetsTrain = []

        noisyValidation = []
        noisySymbolsValidation = []
        targetsValidation = []

        noisyTest = []
        noisySymbolsTest = []
        targetsTest = []

        for sample in train:
            noisyTrain.append([sample["left-noisy"], sample["right-noisy"]])
            noisySymbolsTrain.append([np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["right-noisy"], sample["right-symbol"]))])
            targetsTrain.append(sample["target"])

        for sample in validation:
            noisyValidation.append([sample["left-noisy"], sample["right-noisy"]])
            noisySymbolsValidation.append([np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["right-noisy"], sample["right-symbol"]))])
            targetsValidation.append(sample["target"])

        for sample in test:
            noisyTest.append([sample["left-noisy"], sample["right-noisy"]])
            noisySymbolsTest.append([np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["right-noisy"], sample["right-symbol"]))])
            targetsTest.append(sample["target"])

        return [
            np.array(noisyTrain),
            np.array(noisySymbolsTrain),
            np.array(targetsTrain),
            
            np.array(noisyValidation),
            np.array(noisySymbolsValidation),
            np.array(targetsValidation),
            
            np.array(noisyTest),
            np.array(noisySymbolsTest),
            np.array(targetsTest)
        ]

    def getSequencialDataWithDontCare(self, cross_validation_index, num_validation, num_test):
        
        train, validation, test = self.__splitData(cross_validation_index, num_validation, num_test)

        dontCare = []
        for i in range(784):
            dontCare.append(0.5)
        
        noisyTrain = []
        noisySymbolsTrain = []
        targetsTrain = []

        noisyValidation = []
        noisySymbolsValidation = []
        targetsValidation = []

        noisyTest = []
        noisySymbolsTest = []
        targetsTest = []

        index = 0

        for sample in train:
            dontCare = []
            for i in range(784):
                dontCare.append(random.uniform(0, 1))
            noisyTrain.append([np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["right-noisy"], dontCare))])
            #noisySymbolsTrain.append([np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["right-noisy"], sample["right-symbol"]))])
            if index < 300:
                noisySymbolsTrain.append([np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["right-noisy"], sample["right-symbol"]))])
            else:
                noisySymbolsTrain.append([np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["right-noisy"], dontCare))])
            targetsTrain.append(sample["target"])
            index += 1

        for sample in validation:
            dontCare = []
            for i in range(784):
                dontCare.append(random.uniform(0, 1))
            noisyValidation.append([np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["right-noisy"], dontCare))])
            noisySymbolsValidation.append([np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["right-noisy"], sample["right-symbol"]))])
            targetsValidation.append(sample["target"])

        for sample in test:
            dontCare = []
            for i in range(784):
                dontCare.append(random.uniform(0, 1))
            noisyTest.append([np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["right-noisy"], dontCare))])
            noisySymbolsTest.append([np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["right-noisy"], dontCare))])
            targetsTest.append(sample["target"])

        return [
            np.array(noisyTrain),
            np.array(noisySymbolsTrain),
            np.array(targetsTrain),
            
            np.array(noisyValidation),
            np.array(noisySymbolsValidation),
            np.array(targetsValidation),
            
            np.array(noisyTest),
            np.array(noisySymbolsTest),
            np.array(targetsTest)
        ]

    def getParallelData(self, cross_validation_index, num_validation, num_test):
        
        train, validation, test = self.__splitData(cross_validation_index, num_validation, num_test)
        
        noisyTrain = []
        noisySymbolsTrain = []
        targetsTrain = []

        noisyValidation = []
        noisySymbolsValidation = []
        targetsValidation = []

        noisyTest = []
        noisySymbolsTest = []
        targetsTest = []

        for sample in train:
            noisyTrain.append(np.concatenate((sample["left-noisy"], sample["right-noisy"])))
            noisySymbolsTrain.append(np.concatenate((sample["left-noisy"], sample["left-symbol"], sample["right-noisy"], sample["right-symbol"])))
            targetsTrain.append(sample["target"])

        for sample in validation:
            noisyValidation.append(np.concatenate((sample["left-noisy"], sample["right-noisy"])))
            noisySymbolsValidation.append(np.concatenate((sample["left-noisy"], sample["left-symbol"], sample["right-noisy"], sample["right-symbol"])))
            targetsValidation.append(sample["target"])

        for sample in test:
            noisyTest.append(np.concatenate((sample["left-noisy"], sample["right-noisy"])))
            noisySymbolsTest.append(np.concatenate((sample["left-noisy"], sample["left-symbol"], sample["right-noisy"], sample["right-symbol"])))
            targetsTest.append(sample["target"])

        return [
            np.array(noisyTrain),
            np.array(noisySymbolsTrain),
            np.array(targetsTrain),
            
            np.array(noisyValidation),
            np.array(noisySymbolsValidation),
            np.array(targetsValidation),
            
            np.array(noisyTest),
            np.array(noisySymbolsTest),
            np.array(targetsTest)
        ]

    def getParallelDataWithDontCare(self, cross_validation_index, num_validation, num_test):
        
        train, validation, test = self.__splitData(cross_validation_index, num_validation, num_test)
        
        noisyTrain = []
        noisySymbolsTrain = []
        targetsTrain = []

        noisyValidation = []
        noisySymbolsValidation = []
        targetsValidation = []

        noisyTest = []
        noisySymbolsTest = []
        targetsTest = []

        index = 0

        for sample in train:
            dontCare = []
            for i in range(784):
                dontCare.append(random.uniform(0, 1))
            noisyTrain.append(np.concatenate((sample["left-noisy"], dontCare, sample["right-noisy"], dontCare)))
            if index < 300:
                noisySymbolsTrain.append(np.concatenate((sample["left-noisy"], sample["left-symbol"], sample["right-noisy"], sample["right-symbol"])))
            else:
                noisySymbolsTrain.append(np.concatenate((sample["left-noisy"], dontCare, sample["right-noisy"], dontCare)))
            targetsTrain.append(sample["target"])

            index += 1

        for sample in validation:
            dontCare = []
            for i in range(784):
                dontCare.append(random.uniform(0, 1))
            noisyValidation.append(np.concatenate((sample["left-noisy"], dontCare, sample["right-noisy"], dontCare)))
            noisySymbolsValidation.append(np.concatenate((sample["left-noisy"], sample["left-symbol"], sample["right-noisy"], sample["right-symbol"])))
            targetsValidation.append(sample["target"])

        for sample in test:
            dontCare = []
            for i in range(784):
                dontCare.append(random.uniform(0, 1))
            noisyTest.append(np.concatenate((sample["left-noisy"], dontCare, sample["right-noisy"], dontCare)))
            noisySymbolsTest.append(np.concatenate((sample["left-noisy"], dontCare, sample["right-noisy"], dontCare)))
            targetsTest.append(sample["target"])

        return [
            np.array(noisyTrain),
            np.array(noisySymbolsTrain),
            np.array(targetsTrain),
            
            np.array(noisyValidation),
            np.array(noisySymbolsValidation),
            np.array(targetsValidation),
            
            np.array(noisyTest),
            np.array(noisySymbolsTest),
            np.array(targetsTest)
        ]

    def __loadData(self, folder):
        
        data = np.load(folder + "/data.npy")
        self.data = data.item()

        self.symbolsIn = np.load(folder + "/symbols-in.npy")
        self.symbolsOut = np.load(folder + "/symbols-out.npy")

    def saveData(self, folder):
        
        np.save(folder + "/data", self.data)

        np.save(folder + "/symbols-in", self.symbolsIn)
        np.save(folder + "/symbols-out", self.symbolsOut)