from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

class FeedForwardModel(object):
    
    def __init__(self, hidden_layers, n_inputs, n_outputs):
        
        # Model Parameters
        self.hidden_layers = hidden_layers # array holding the number of units in each hidden layer
        self.n_inputs = n_inputs # number of input units
        self.n_outputs = n_outputs # number of output units

        # Internal
        self._trained = False

        self.__create_model()

    def __create_model(self):
        self.model = Sequential()
        for index, n_units in enumerate(self.hidden_layers):
            if index == 0:
                self.model.add(Dense(n_units, input_dim=self.n_inputs, activation='relu'))
        self.model.add(Dense(self.n_outputs, activation='sigmoid'))

        self.model.compile(loss='mean_squared_error',
                      optimizer='Adam',
                      metrics=['accuracy'])

    def train(self, x, y, val_x, val_y, batch_size, epochs):

        checkpointCallback = ModelCheckpoint("checkpoint.hdf5")
        self.model.fit(x, y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=epochs, shuffle=False, callbacks=[checkpointCallback])

        self.model = load_model("checkpoint.hdf5")

        self._trained = True
            

    def save(self, file):
        if self._trained == False:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        self.model.save(file)

    def evaluate(self, x, y):
        if self._trained == False:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        return self.model.evaluate(x, y)