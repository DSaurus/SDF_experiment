import numpy as np
import math
import matplotlib.pyplot as plt

def int_sin(l, r, a):
    if a != 0:
        return - 1 / a *( np.cos(a*r) - np.cos(a*l))
    return 0

def int_cos(l, r, a):
    if a != 0:
        return 1 / a * ( np.sin(a*r) - np.sin(a*l))
    return (r-l)

def fourier(an, bn, x):
    y = np.zeros_like(x)
    y += an[0] / 2
    pi = math.acos(-1)
    for n in range(1, len(an)):
        y += an[n] * np.cos(pi*n*x) + bn[n] * np.sin(pi*n*x)
    return y

def position_encoding(x, dims):
    r = 0.005
    pi = math.acos(-1)
    an = [int_cos(x-r, x+r, n*pi) for n in range(dims)]
    bn = [int_sin(x-r, x+r, n*pi) for n in range(dims)]

    # x = np.linspace(-1, 1, 1000)
    # y = fourier(an, bn, x)
    # plt.plot(x, y)
    # plt.savefig('fourier.jpg')
    an = np.array(an)
    bn = np.array(bn)
    return np.concatenate((an, bn))

def int_sin_batch(l, r, a):
    if a != 0:
        return - 1 / a * (np.cos(a*r) - np.cos(a*l))
    return np.zeros_like(l)

def int_cos_batch(l, r, a):
    if a != 0:
        return 1 / a * ( np.sin(a*r) - np.sin(a*l))
    res = np.zeros_like(l)
    res = (r-l)
    return res

def position_encoding_batch(x, dims):
    r = 0.005
    pi = math.acos(-1)
    an = [int_cos_batch(x-r, x+r, n*pi) for n in range(dims)]
    bn = [int_sin_batch(x-r, x+r, n*pi) for n in range(dims)]

    an = np.transpose(np.array(an))
    bn = np.transpose(np.array(bn))
    return np.concatenate((an, bn), axis=1)

if __name__ == "__main__":
    x = np.array([1, 0.5, 0.2])
    res = position_encoding_batch(x, 64)
    print(res.shape)
    print(res)

