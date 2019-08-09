from __future__ import division, print_function, absolute_import

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

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
                self.model.add(LSTM(n_units))
        self.model.add(Dense(self.n_outputs, activation='sigmoid'))

        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

    def train(self, x, y, batch_size, epochs):

        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=False)

        self._trained = True
            

    def save(file):
        if self._trained == False and model == None:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        self.model.save(file)

    def evaluate(self, x, y):
        if self._trained == False and model == None:
            errmsg = "[!] Error: model not trained."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        return model.evaluate(x, y)