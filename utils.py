import random


def roundArray(array, digits=3):
    out = []
    for el in array:
        out.append(round(el, digits))
    return out


def randArray(size):
    out = []
    for i in range(0, size):
        out.append((random.random()-0.5)*2)
    return out
