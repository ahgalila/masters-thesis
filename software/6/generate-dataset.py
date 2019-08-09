import sys
sys.dont_write_bytecode = True

#from data import DataLoader
#from data_explicit import DataLoader
from test_independant import TestDataLoader

#dataLoader = DataLoader(sample_size=20)
#dataLoader.saveData("MNIST_data/dataset")

testDataLoader = TestDataLoader()
testDataLoader.saveData("MNIST_data/dataset")