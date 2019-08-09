from PIL import Image, ImageDraw, ImageFont
from lstm_model import LSTMModel
import numpy as np

def one_hot(value):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    result[value] = 1
    return result

def argmax(one_hot):
    max = 0
    max_index = 0
    for i in range(len(one_hot)):
        if one_hot[i] > max:
            max = one_hot[i]
            max_index = i
    return max_index

def renderActivations(draw, activations, startX, startY):
    for index in range(len(activations)):
        intensity = int(((activations[index] + 1) / 2.0) * 255)
        for x in range(startX + index * 14, startX + index * 14 + 14):
            for y in range(startY, startY + 14):
                draw.point((x, y), fill=(intensity, intensity, intensity))

def renderImage(image, name, startX, startY):
    target = Image.open(name, 'r')
    image.paste(target, (startX, startY))

def renderResults(name, results, test_x, activations_0, activations_1):
    
    image = Image.new('RGB', (28 * 3 + 280, len(results) * 160))
    draw = ImageDraw.Draw(image)

    for index, result in enumerate(results):

        renderImage(image, "MNIST_data/" + str(argmax(test_x[index][0])) + ".png", 0 * 28, index * 160 + 0 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[0])) + ".png", 2 * 28, index * 160 + 0 * 32)
        renderActivations(draw, activations_1[index][0], 3 * 28, index * 160 + 0 * 32)
        renderActivations(draw, activations_0[index][0], 3 * 28, index * 160 + 0 * 32 + 14)

        renderImage(image, "MNIST_data/" + str(argmax(test_x[index][1])) + ".png", 0 * 28, index * 160 + 1 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[1])) + ".png", 2 * 28, index * 160 + 1 * 32)
        renderActivations(draw, activations_1[index][1], 3 * 28, index * 160 + 1 * 32)
        renderActivations(draw, activations_0[index][1], 3 * 28, index * 160 + 1 * 32 + 14)

        renderImage(image, "MNIST_data/" + "plus.png", 0 * 28, index * 160 + 2 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[2])) + ".png", 2 * 28, index * 160 + 2 * 32)
        renderActivations(draw, activations_1[index][2], 3 * 28, index * 160 + 2 * 32)
        renderActivations(draw, activations_0[index][2], 3 * 28, index * 160 + 2 * 32 + 14)

        renderImage(image, "MNIST_data/" + "equals.png", 0 * 28, index * 160 + 3 * 32)
        renderImage(image, "MNIST_data/" + str(argmax(result[3])) + ".png", 2 * 28, index * 160 + 3 * 32)
        renderActivations(draw, activations_1[index][3], 3 * 28, index * 160 + 3 * 32)
        renderActivations(draw, activations_0[index][3], 3 * 28, index * 160 + 3 * 32 + 14)

    image.save(name  + ".png")

plus_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
equals_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
dummy = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

x = []
y = []

test_x = []
test_y = []

testWatchListA = np.load("MNIST_data/unseen_a.npy").item()
testWatchListB = np.load("MNIST_data/unseen_b.npy").item()

for a in range(10):
    for b in range(10):
        if a == 3:
            test_x.append([one_hot(b) + [1, 0, 0], one_hot(a) + [1, 0, 0], plus_vector, equals_vector])
            test_y.append([one_hot(b), one_hot(a), one_hot((a + b) % 10), one_hot((a + b) // 10)])
        else:
            x.append([one_hot(b) + [1, 0, 0], one_hot(a) + [1, 0, 0], plus_vector, equals_vector])
            y.append([one_hot(b), one_hot(a), one_hot((a + b) % 10), one_hot((a + b) // 10)])

x = np.array(x)
y = np.array(y)

test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [20, 20], 13, 10)
model.train(x, y, x, y, 10, 5000)

results = model.predict(test_x)
activations_0 = model.getActivations(0, test_x)
activations_1 = model.getActivations(1, test_x)

count = 0

for index, result in enumerate(results):
    right = argmax(result[2])
    left = argmax(result[3])
    rightTarget = argmax(test_y[index][2])
    leftTarget = argmax(test_y[index][3])

    if left == leftTarget and right == rightTarget:
        count += 1

renderResults("one-hot", results, test_x, activations_0, activations_1)

print("SCORE: " + str(float(count) / len(results)))