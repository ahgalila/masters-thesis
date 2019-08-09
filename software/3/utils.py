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

def renderImage(image, name, startX, startY):
    target = Image.open(name, 'r')
    image.paste(target, (startX, startY))

def renderBlock(image, draw, indexX, indexY, left, right, result_left, result_right):
    renderDigit(draw, left, indexX * 28 * 7, indexY * 28)
    renderImage(image, "MNIST_data/plus.png", indexX * 28 * 7 + 28, indexY * 28)
    renderDigit(draw, right, indexX * 28 * 7 + 56, indexY * 28)
    renderImage(image, "MNIST_data/equals.png", indexX * 28 * 7 + 84, indexY * 28)
    renderImage(image, "MNIST_data/" + str(result_left) + ".png", indexX * 28 * 7 + 112, indexY * 28)
    renderImage(image, "MNIST_data/" + str(result_right) + ".png", indexX * 28 * 7 + 140, indexY * 28)

def renderResults(inputs, outputs):
    image = Image.new('RGB', (28 * 7 * 10 - 28, 28 * 3 * 10))
    draw = ImageDraw.Draw(image)
    index = 0
    for a in range(10):
        for b in range(10):
            renderBlock(image, draw, a, b * 3, inputs[index][0], inputs[index][1], argmax(outputs[index][:10]), argmax(outputs[index][10:]))
            renderBlock(image, draw, a, b * 3 + 1, inputs[index + 1][0], inputs[index + 1][1], argmax(outputs[index + 1][:10]), argmax(outputs[index + 1][10:]))
            index += 2
    image.save("results.png")
