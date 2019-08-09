from PIL import Image, ImageDraw, ImageFont
import numpy as np

digit = "+"

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
        if r == 255:
            data.append(1.0)
        else:
            data.append(0.0)

np.save("MNIST_data/plus", data)

image.show()