import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import LSTM, Dense, Input, concatenate
from keras.datasets import mnist

# Some parameters
hidden_size = 5
input_length = 2

# temperature encoding
def temperature(nr):
    return np.concatenate([np.ones(nr), np.zeros(9-nr)])

def temperature_output(nr):
    if nr < 10:
        return np.concatenate([temperature(0), temperature(nr)])
    else:
        return np.concatenate([temperature(1), temperature(nr-10)])

def get_ordered_data(images, labels):
    ordered_images = 10*[[]]
    for i in range(len(images)):
        ordered_images[labels[i]].append(images[i])
    return ordered_images

def get_symbolic_sequences(N, input_length, min_input, max_input):
    # get N symbolic sequences
    X = np.random.randint(min_input, max_input+1, size=(N, input_length))
    Y = X.sum(axis=1)
    return X, Y

def get_sequences(images_list, N, input_length, min_input, max_input, use_symbols):

    def get_image(i):
        im = images_list[i][np.random.randint(len(images_list))]
        return im

    # get sequences on symbolic level 
    X_symb, Y_symb = get_symbolic_sequences(N, input_length, min_input, max_input)

    # create input_data
    X = np.array([[get_image(x) for x in seq] for seq in X_symb])
    Y = np.array([temperature_output(y) for y in Y_symb])
    input_symbols = np.array([[temperature(x) for x in seq] for seq in X_symb])

    # reshape
    shape = X.shape
    X_reshape = X.reshape((shape[0], shape[1], shape[2]*shape[3]))

    if not use_symbols: 
        input_symbols = np.zeros_like(input_symbols)

    return X_reshape, input_symbols, Y
    

def get_data(images, labels, N, min_input=0, max_input=9, input_length=2, use_symbols=False):
    # order data
    images_list = get_ordered_data(images, labels)
    X, X_symb, Y = get_sequences(images_list, N, input_length, min_input, max_input, use_symbols)
    return X, X_symb, Y


def model(hidden_size):
    images = Input(shape=(input_length, 28*28), name='input_images')
    symbols = Input(shape=(input_length, 9), name='input_symbols')
    concat_layer = concatenate([images, symbols], axis=-1)
    recurrent_layer = LSTM(hidden_size, name='hidden1', trainable=True, return_sequences=True)(concat_layer)
    recurrent_layer2 = LSTM(hidden_size, name='hidden2', trainable=True)(recurrent_layer)
    output_layer = Dense(18, activation='sigmoid')(recurrent_layer2)

    model = Model(inputs=[images, symbols], outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', 'mean_squared_error'])

    return model

if __name__ == '__main__':
    np.random.seed(10)
    use_symbols = False      # whether or not to use symbols
    epochs = 200
    hidden_size = 50
    # load data
    (mnist_trainX, mnist_trainY), (mnist_testX, mnist_testY) = mnist.load_data()
    # transform to training data
    X_train, X_symb, Y_train = get_data(mnist_trainX, mnist_trainY, 1000, use_symbols=use_symbols)
    X_test, X_test_symb, Y_test = get_data(mnist_testX, mnist_testY, 1000, use_symbols=use_symbols)

    # X_train = np.zeros_like(X_train)
    # X_test = np.zeros_like(X_test)

    # save to file
    # np.savetxt('inputs.txt', X_test_symb, delimiter=',', fmt='%5s')
    # np.savetxt('targets.txt', Y_test, delimiter=',')

    m = model(hidden_size=hidden_size)

    # train_model
    m.fit({'input_images': X_train, 'input_symbols': X_symb}, Y_train, epochs=epochs, shuffle=True, validation_split=0.2)

    print ('\n\n\n', m.evaluate({'input_images': X_test, 'input_symbols':X_test_symb}, Y_test))
    predictions = m.predict({'input_images': X_test, 'input_symbols':X_test_symb})

    # np.savetxt('predictions.txt', predictions, delimiter=',')

