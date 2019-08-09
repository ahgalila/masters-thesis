from PIL import Image, ImageDraw, ImageFont
import numpy as np

digit = ":"

font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)

image = Image.new('RGB', (28, 28))
draw = ImageDraw.Draw(image)

(width, height) = draw.textsize(digit, font = font)

x = 14 - width / 2
y = 14 - height / 2

draw.text((x, y), digit, font = font, fill = (255,255,255,255))

data = []
for row in range(28):
    for col in range(28):
        r, g, b = image.getpixel((col, row))
        intensity = r / 255.0
        data.append(intensity)

np.save("MNIST_data/colon", data)
image.save("MNIST_data/colon.png")

'''digit = np.load("MNIST_data/equals.npy")
xd = 0
for x in range(28):
    yd = 0
    for y in range(28):
        intensity = int(digit[yd * 28 + xd] * 255)
        draw.point((x, y), fill=(intensity, intensity, intensity))
        yd += 1
    xd += 1'''

image.show()