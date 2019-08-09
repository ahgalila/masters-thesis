from __future__ import division, print_function, absolute_import
from PIL import Image, ImageDraw, ImageFont

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

def renderDigit(draw, digit, startX, startY):
    xd = 0
    for x in range(startX, startX + 28):
        yd = 0
        for y in range(startY, startY + 28):
            intensity = int(digit[yd * 28 + xd] * 255)
            draw.point((x, y), fill=(intensity, intensity, intensity))
            yd += 1
        xd += 1

def renderActivations(draw, activations, startX, startY):
    for unitX in range(64):
        for unitY in range(4):
            index = unitY * 64 + unitX
            intensity = int(((activations[index] + 1) / 2.0) * 255)
            for x in range(startX + unitX * 4, startX + unitX * 4 + 4):
                for y in range(startY + unitY * 4, startY + unitY * 4 + 4):
                    draw.point((x, y), fill=(intensity, intensity, intensity))

def renderActivations200(draw, activations, startX, startY):
    for unitX in range(100):
        for unitY in range(2):
            index = unitY * 100 + unitX
            intensity = int(((activations[index] + 1) / 2.0) * 255)
            for x in range(startX + unitX * 8, startX + unitX * 8 + 8):
                for y in range(startY + unitY * 8, startY + unitY * 8 + 8):
                    draw.point((x, y), fill=(intensity, intensity, intensity))

def renderActivationsBig(draw, activations, startX, startY):
    for index in range(10):
        intensity = int(((activations[index] + 1) / 2.0) * 255)
        for x in range(startX + index * 28, startX + index * 28 + 28):
            for y in range(startY, startY + 28):
                draw.point((x, y), fill=(intensity, intensity, intensity))

def renderImage(image, name, startX, startY):
    target = Image.open(name, 'r')
    image.paste(target, (startX, startY))

def renderBlock(image, draw, indexX, indexY, left, right, class_left, class_right, result_left, result_right):
    renderDigit(draw, left, indexX * 28 * 11, indexY * 28)
    renderImage(image, "MNIST_data/plus.png", indexX * 28 * 11 + 28, indexY * 28)
    renderDigit(draw, right, indexX * 28 * 11 + 56, indexY * 28)
    renderImage(image, "MNIST_data/implies.png", indexX * 28 * 11 + 84, indexY * 28)
    renderImage(image, "MNIST_data/" + str(class_left) + ".png", indexX * 28 * 11 + 112, indexY * 28)
    renderImage(image, "MNIST_data/plus.png", indexX * 28 * 11 + 140, indexY * 28)
    renderImage(image, "MNIST_data/" + str(class_right) + ".png", indexX * 28 * 11 + 168, indexY * 28)
    renderImage(image, "MNIST_data/equals.png", indexX * 28 * 11 + 196, indexY * 28)
    renderImage(image, "MNIST_data/" + str(result_left) + ".png", indexX * 28 * 11 + 224, indexY * 28)
    renderImage(image, "MNIST_data/" + str(result_right) + ".png", indexX * 28 * 11 + 252, indexY * 28)

def renderResults(file, inputs, outputs):
    image = Image.new('RGB', (28 * 11 * 10 - 28, 28 * 3 * 10))
    draw = ImageDraw.Draw(image)
    index = 0
    for a in range(10):
        for b in range(10):
            renderBlock(image, draw, a, b * 3, inputs[index][1], inputs[index][0], argmax(outputs[index][1][10:]), argmax(outputs[index][0][10:]), argmax(outputs[index][2][:10]), argmax(outputs[index][2][10:]))
            renderBlock(image, draw, a, b * 3 + 1, inputs[index + 1][1], inputs[index + 1][0], argmax(outputs[index + 1][1][10:]), argmax(outputs[index + 1][0][10:]), argmax(outputs[index + 1][2][:10]), argmax(outputs[index + 1][2][10:]))
            index += 2
    image.save(file)

def getOperator(input):
    value = argmax(input)
    if value == 0:
        return "plus"
    if value == 1:
        return "times"
    if value == 2:
        return "minus"
    if value == 3:
        return "divide"

def getOperatorSign(input):
    value = argmax(input)
    if value == 0:
        return "+"
    if value == 1:
        return "x"
    if value == 2:
        return "-"
    if value == 3:
        return "/"
    return "::"