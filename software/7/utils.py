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

