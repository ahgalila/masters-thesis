from __future__ import division, print_function, absolute_import

import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
            if index == 0:
                self.model.add(LSTM(n_units, return_sequences=True, input_shape=(self.time_steps, self.n_inputs)))
            else:
                self.model.add(LSTM(n_units, return_sequences=True))
        self.model.add(Dense(self.n_outputs, activation='sigmoid'))

        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self, x, y, val_x, val_y, batch_size, epochs):

        checkpointCallback = ModelCheckpoint("checkpoint.hdf5")
        earlyStoppingCallback = EarlyStopping(patience=10)
        result = self.model.fit(x, y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=epochs, shuffle=False, callbacks=[checkpointCallback, earlyStoppingCallback])
        history = result.history["val_loss"]
        '''history_train = result.history["loss"]
        print(history)
        print(history_train)
        plt.plot(range(len(history)), history_train, "o", markersize=5, label="training loss")
        plt.plot(range(len(history)), history, "s", markersize=5, label="validation loss")
        plt.legend()
        plt.show()'''

        min_value = history[0]
        min_index = 0
        for index, val_loss in enumerate(history):
            if val_loss < min_value:
                min_value = val_loss
                min_index = index

        self.model = load_model("checkpoint.hdf5")

        self._trained = True

        return min_index
            

    def save(self, file):
        if self._trained == False:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        self.model.save(file)

    def load(self, file):
        self.model = load_model(file)
        self._trained = True

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

    def getActivations(self, layer, x):
        if self._trained == False:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        model = Sequential()
        for index, n_units in enumerate(self.hidden_layers):
            if index <= layer:
                if index == 0:
                    model.add(LSTM(n_units, return_sequences=True, input_shape=(self.time_steps, self.n_inputs)))
                else:
                    model.add(LSTM(n_units, return_sequences=True))
                model.set_weights(self.model.get_weights())

        model.compile(loss='mean_squared_error',
                      optimizer='adam')

        return model.predict(x)