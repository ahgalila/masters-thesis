from PIL import Image, ImageDraw, ImageFont
from lstm_model import LSTMModel
import numpy as np

def temperature(value):
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = -1
    while index >= -value:
        result[index] = 1
        index -= 1
    return result

def is_temperature(value, target):
    for i in range(len(value)):
        if (target[i] == 1 and value[i] < 0.5) or target[i] == 0 and value[i] > 0.5:
            return False
    return True

def temperatureToInt(temp):
    for i in range(10):
        if is_temperature(temp, temperature(i)):
            return i
    return 0

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

        renderImage(image, "MNIST_data/" + str(temperatureToInt(test_x[index][0][:9])) + ".png", 0 * 28, index * 160 + 0 * 32)
        renderImage(image, "MNIST_data/" + str(temperatureToInt(result[0])) + ".png", 2 * 28, index * 160 + 0 * 32)
        renderActivations(draw, activations_1[index][0], 3 * 28, index * 160 + 0 * 32)
        renderActivations(draw, activations_0[index][0], 3 * 28, index * 160 + 0 * 32 + 14)

        renderImage(image, "MNIST_data/" + str(temperatureToInt(test_x[index][1][:9])) + ".png", 0 * 28, index * 160 + 1 * 32)
        renderImage(image, "MNIST_data/" + str(temperatureToInt(result[1])) + ".png", 2 * 28, index * 160 + 1 * 32)
        renderActivations(draw, activations_1[index][1], 3 * 28, index * 160 + 1 * 32)
        renderActivations(draw, activations_0[index][1], 3 * 28, index * 160 + 1 * 32 + 14)

        renderImage(image, "MNIST_data/" + "plus.png", 0 * 28, index * 160 + 2 * 32)
        renderImage(image, "MNIST_data/" + str(temperatureToInt(result[2])) + ".png", 2 * 28, index * 160 + 2 * 32)
        renderActivations(draw, activations_1[index][2], 3 * 28, index * 160 + 2 * 32)
        renderActivations(draw, activations_0[index][2], 3 * 28, index * 160 + 2 * 32 + 14)

        renderImage(image, "MNIST_data/" + "equals.png", 0 * 28, index * 160 + 3 * 32)
        renderImage(image, "MNIST_data/" + str(temperatureToInt(result[3])) + ".png", 2 * 28, index * 160 + 3 * 32)
        renderActivations(draw, activations_1[index][3], 3 * 28, index * 160 + 3 * 32)
        renderActivations(draw, activations_0[index][3], 3 * 28, index * 160 + 3 * 32 + 14)

    image.save(name  + ".png")

plus_vector = [0, 0, 1, 0]
equals_vector = [0, 0, 0, 1]

x = []
y = []

test_x = []
test_y = []

testWatchListA = np.load("MNIST_data/unseen_a.npy").item()
testWatchListB = np.load("MNIST_data/unseen_b.npy").item()

for a in range(10):
    for b in range(10):
        if a == 3:
            test_x.append([temperature(b) + [1, 0, 0], temperature(a) + [1, 0, 0], temperature(0) + [0, 1, 0], temperature(0) + [0, 0, 1]])
            test_y.append([temperature(b), temperature(a), temperature((a + b) % 10), temperature((a + b) // 10)])
        else:
            x.append([temperature(b) + [1, 0, 0], temperature(a) + [1, 0, 0], temperature(0) + [0, 1, 0], temperature(0) + [0, 0, 1]])
            y.append([temperature(b), temperature(a), temperature((a + b) % 10), temperature((a + b) // 10)])

x = np.array(x)
y = np.array(y)

test_x = np.array(test_x)
test_y = np.array(test_y)

model = LSTMModel(4, [20, 20], 12, 9)
model.train(x, y, x, y, 10, 5000)

results = model.predict(test_x)
activations_0 = model.getActivations(0, test_x)
activations_1 = model.getActivations(1, test_x)

count = 0

for index, result in enumerate(results):
    right = temperatureToInt(result[2])
    left = temperatureToInt(result[3])
    rightTarget = temperatureToInt(test_y[index][2])
    leftTarget = temperatureToInt(test_y[index][3])

    if left == leftTarget and right == rightTarget:
        count += 1

renderResults("temperature", results, test_x, activations_0, activations_1)

print("SCORE: " + str(float(count) / len(results)))