from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

class LSTMModel(object):
    
    def __init__(self, time_steps, hidden_layers, n_inputs, n_outputs):
        
        # Model Parameters
        self.time_steps = time_steps # timesteps to unroll
        self.hidden_layers = hidden_layers # hidden LSTM units
        self.n_inputs = n_inputs # number of input units
        self.n_outputs = n_outputs # number of output units

        # Internal
        self._trained = False

        self.__create_model()

    def __create_model(self):
        self.model = Sequential()
        for index, n_units in enumerate(self.hidden_layers):
            return_sequences = True
            if index == len(self.hidden_layers) - 1:
                return_sequences = False
            if index == 0:
                self.model.add(LSTM(n_units, return_sequences=return_sequences, input_shape=(self.time_steps, self.n_inputs)))
            else:
                self.model.add(LSTM(n_units, return_sequences=return_sequences))
        self.model.add(Dense(self.n_outputs, activation='sigmoid'))

        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
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

    def predict(self, x):
        if self._trained == False:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        return self.model.predict(x)