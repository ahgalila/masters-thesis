from __future__ import division, print_function, absolute_import
import sys

sys.dont_write_bytecode = True

import data

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import numpy as np

class LSTMModel(object):
    def __init__(self):
        # Model Parameters
        self.time_steps=2 # timesteps to unroll
        self.n_units=512 # hidden LSTM units
        self.n_inputs=784 # single mnist img (an mnist img is 28x28)
        self.n_outputs=20 # 2 one hot vectors
        self.batch_size=100 # Size of each batch
        self.n_epochs=100
        # Internal
        self._data_loaded = False
        self._trained = False

    def __create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.n_units, return_sequences=True, input_shape=(self.time_steps, self.n_inputs)))
        self.model.add(LSTM(self.n_units))
        self.model.add(Dense(self.n_outputs, activation='sigmoid'))

        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

    def __load_data(self):
        [
            self.trainData,
            self.trainDataSymbols,
            self.trainTargets,
            self.trainTargetsSymbols,
            self.trainTargetClasses,
            self.testData,
            self.testDataSymbols,
            self.testTargets,
            self.testTargetsSymbols,
            self.validationData,
            self.validationDataSymbols,
            self.validationTargets,
            self.validationTargetsSymbols,
            self.validationTargetClasses 
        ] = data.loadDataSet()
        
        self._data_loaded = True

    def train(self, save_model=False):
        self.__create_model()
        if self._data_loaded == False:
            self.__load_data()

        x_train = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.trainData]
        x_train = np.array(x_train).reshape((-1, self.time_steps, self.n_inputs))

        self.model.fit(x_train, self.trainTargetsSymbols,
                  batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

        self._trained = True
        
        if save_model:
            self.model.save("./lstm-model.h5")

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        x_test = [x.reshape((-1, self.time_steps, self.n_inputs)) for x in self.testData]
        x_test = np.array(x_test).reshape((-1, self.time_steps, self.n_inputs))

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(x_test, self.testTargetsSymbols)
        print(test_loss)


if __name__ == "__main__":
    lstm_classifier = LSTMModel()
    lstm_classifier.train(save_model=False)
    lstm_classifier.evaluate()