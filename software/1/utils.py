from __future__ import division, print_function, absolute_import

import datetime, json, requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from random import randint

def argmax(one_hot):
    max = 0
    max_index = 0
    for i in range(len(one_hot)):
        if one_hot[i] > max:
            max = one_hot[i]
            max_index = i
    return max_index

def one_hot(input):
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    output[input] = 1
    return output

def plot_performance(train_losses, validation_losses, num_epochs, title, name, show = False):
    plt.plot(train_losses, label = "Train")
    plt.plot(validation_losses, label = "Validation")
    plt.xticks(range(0, num_epochs, num_epochs // 10))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig("plots/" + name + "-" + ".png")
    if show:
        plt.show()

def render_digit(digit, xOrigin, yOrigin, draw):
    for y in range(28):
        for x in range(28):
            intensity = int(digit[y * 28 + x] * 255)
            draw.point((xOrigin + x, yOrigin + y), fill = (intensity, intensity, intensity))

def render_row(input, output, x, y, operatorWidth, equalsWidth, colonWidth, draw, font, target = []):
    render_digit(input[:784], x, y, draw)
    draw.text((x + 28, y), "+", font = font, fill = (255,255,255,255))
    render_digit(input[784:], x + 28 + operatorWidth, y, draw)
    draw.text((x + 28 + operatorWidth + 28, y), "=", font = font, fill = (255,255,255,255))
    render_digit(output[:784], x + 28 + operatorWidth + 28 + equalsWidth, y, draw)
    render_digit(output[784:], x + 28 + operatorWidth + 28 + equalsWidth + 28, y, draw)
    if len(target) > 0:
        draw.text((x + 28 + operatorWidth + 28 + equalsWidth + 28 + 28, y), ":", font = font, fill = (255,255,255,255))
        render_digit(target[:784], x + 28 + operatorWidth + 28 + equalsWidth + 28 + 28 + colonWidth, y, draw)
        render_digit(target[784:], x + 28 + operatorWidth + 28 + equalsWidth + 28 + 28 + colonWidth + 28, y, draw)

def render_output(validationOutput, testOutput, validationData, testData, validationTargets, name, show = False):
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 30)
    image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(image)
    (operatorWidth, operatorHeight) = draw.textsize('+', font = font)
    (equalsWidth, equalsHeight) = draw.textsize('=', font = font)
    (colonWidth, colonHeight) = draw.textsize(':', font = font)
    width = 28 + operatorWidth + 28 + equalsWidth + 28 + 28
    width *= 2
    width += colonWidth + 28 + 28
    height = len(validationOutput) * 28
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    for i in range(len(validationOutput)):
        render_row(validationData[i], validationOutput[i], 0, i * 28, operatorWidth, equalsWidth, colonWidth, draw, font, validationTargets[i])
        render_row(testData[i], testOutput[i], 28 + operatorWidth + 28 + equalsWidth + 28 + 28 + colonWidth + 28 + 28, i * 28, operatorWidth, equalsWidth, colonWidth, draw, font)
    if show:
        image.show()
    image.save("outputs/" + name + "-" + ".png")

def renderUnsupervisedData(data):
    num = len(data) // 10
    image = Image.new('RGB', (28 * num, 280))
    draw = ImageDraw.Draw(image)
    for i in range(10):
        for j in range(num):
            render_digit(data[i * 10 + j], j * 28, i * 28, draw)
    image.save("MNIST_data/unsupervised-data.png")
        

def validate(testOutput, testTargets):
    testSuccessCount = 0
    index = 0
    results = []
    for testDigit in testOutput:
        payload = {'x': [testDigit[784:1568].tolist()]}
        r = requests.post('http://localhost:5000/api/predict', data = json.dumps(payload))
        response = r.json()
        results.append(argmax(response['y'][0]))
        if argmax(response['y'][0]) == testTargets[index]:
            testSuccessCount = testSuccessCount + 1
        index = index + 1
    return [testSuccessCount / len(testTargets), results]

def validateSymbols(testOutput, testTargets):
    testSuccessCount = 0
    index = 0
    for testDigit in testOutput:
        output = (argmax(testDigit[1568:1578]) * 10) + argmax(testDigit[1578:])
        target = (argmax(testTargets[index][:10]) * 10) + argmax(testTargets[index][10:])
        if output == target:
            testSuccessCount = testSuccessCount + 1
        index = index + 1
    return testSuccessCount / len(testTargets)

def selectMinBatch(inputs, outputs, size = 100):
    xs = []
    ys = []
    for i in range(size):
        index = randint(0, len(inputs) - 1)
        xs.append(inputs[index])
        ys.append(outputs[index])
    return [xs, ys]

def selectNextMinBatch(inputs, outputs, batch, size = 100):
    start = batch * size
    end = batch * size + size
    return [inputs[start:end], outputs[start:end]]