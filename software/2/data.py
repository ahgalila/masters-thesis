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

        plus = np.load("MNIST_data/plus.npy")

        empty_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(len(mnist.train.images)):
            trainSamples[argmax(mnist.train.labels[i])].append(mnist.train.images[i])
        for i in range(len(mnist.test.images)):
            testSamples[argmax(mnist.test.labels[i])].append(mnist.test.images[i])

        for a in range(0, 10):
            for b in range(0, 10):
                
                self.data[str(a) + "+" + str(b)] = []
            
                leftSymbol = symbols[a]
                rightSymbol = symbols[b]

                target = [empty_one_hot + one_hot(b), empty_one_hot + one_hot(a), one_hot((a + b) // 10) + one_hot((a + b) % 10)]

                symbolsIn.append([rightSymbol, leftSymbol, plus])
                symbolsOut.append(target)

                for i in range(sample_size):
                    left = trainSamples[a][randint(0, len(trainSamples[a]) - 1)]
                    right = trainSamples[b][randint(0, len(trainSamples[b]) - 1)]

                    self.data[str(a) + "+" + str(b)].append({
                        "left-noisy": left,
                        "left-symbol": leftSymbol,
                        "right-noisy": right,
                        "right-symbol": rightSymbol,
                        "plus": plus,
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
            noisyTrain.append([sample["right-noisy"], sample["left-noisy"], sample["plus"]])
            noisySymbolsTrain.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
            targetsTrain.append(sample["target"])

        for sample in validation:
            noisyValidation.append([sample["right-noisy"], sample["left-noisy"], sample["plus"]])
            noisySymbolsValidation.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
            targetsValidation.append(sample["target"])

        for sample in test:
            noisyTest.append([sample["right-noisy"], sample["left-noisy"], sample["plus"]])
            noisySymbolsTest.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
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

    def getSequencialDataNoClassification(self, cross_validation_index, num_validation, num_test):
        
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

        dontCare = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        for sample in train:
            noisyTrain.append([sample["right-noisy"], sample["left-noisy"], sample["plus"]])
            noisySymbolsTrain.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
            targetsTrain.append([dontCare, dontCare, sample["target"][2]])

        for sample in validation:
            noisyValidation.append([sample["right-noisy"], sample["left-noisy"], sample["plus"]])
            noisySymbolsValidation.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
            targetsValidation.append([dontCare, dontCare, sample["target"][2]])

        for sample in test:
            noisyTest.append([sample["right-noisy"], sample["left-noisy"], sample["plus"]])
            noisySymbolsTest.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
            targetsTest.append([dontCare, dontCare, sample["target"][2]])

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
        
        noisyTrain = []
        noisySymbolsTrain = []
        targetsTrain = []

        noisyValidation = []
        noisySymbolsValidation = []
        targetsValidation = []

        noisyTest = []
        noisySymbolsTest = []
        targetsTest = []

        dontCare = []
        for i in range(784):
            dontCare.append(1.0)

        for index, sample in enumerate(train):
            noisyTrain.append([np.concatenate((sample["right-noisy"], dontCare)), np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["plus"], sample["plus"]))])
            if index < 400:
                noisySymbolsTrain.append([np.concatenate((sample["right-noisy"], sample["right-symbol"])), np.concatenate((sample["left-noisy"], sample["left-symbol"])), np.concatenate((sample["plus"], sample["plus"]))])
            else:
                noisySymbolsTrain.append([np.concatenate((sample["right-noisy"], dontCare)), np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["plus"], sample["plus"]))])
            targetsTrain.append(sample["target"])

        for sample in validation:
            noisyValidation.append([np.concatenate((sample["right-noisy"], dontCare)), np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["plus"], sample["plus"]))])
            noisySymbolsValidation.append([np.concatenate((sample["right-noisy"], dontCare)), np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["plus"], sample["plus"]))])
            targetsValidation.append(sample["target"])

        for sample in test:
            noisyTest.append([np.concatenate((sample["right-noisy"], dontCare)), np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["plus"], sample["plus"]))])
            noisySymbolsTest.append([np.concatenate((sample["right-noisy"], dontCare)), np.concatenate((sample["left-noisy"], dontCare)), np.concatenate((sample["plus"], sample["plus"]))])
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