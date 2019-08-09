from PIL import Image, ImageDraw, ImageFont
import numpy as np

digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)

image = Image.new('RGB', (280, 28))
draw = ImageDraw.Draw(image)

for index, digit in enumerate(digits):

    (width, height) = draw.textsize(digit, font = font)

    x = 14 - width / 2
    y = 14 - height / 2

    draw.text((x + index * 28, y), digit, font = font, fill = (255,255,255,255))

image.save("noisy.png")

image.show()