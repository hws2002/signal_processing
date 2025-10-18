import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os
import imageio
from matplotlib.pyplot import cm


import argparse
from functools import lru_cache
PI = math.pi

parser = argparse.ArgumentParser()

parser.add_argument('--N_Fourier', type=int, default=32, help='Number of Fourier Series')
args = parser.parse_args()

# TODO: 1. Change N_Fourier to 2, 4, 8, 16, 32, 64, 128, get visualization results with differnet number of Fourier Series
N_Fourier = args.N_Fourier

# TODO: optional, implement visualization for semi-circle
signal_name = "semicircle"

# TODO: 2. Please implement the function that calculates the Nth fourier coefficient
# Note that n starts from 0
# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am.
 
@lru_cache(maxsize=None)
def square_wave(t):
    return 0.5 * np.sign(math.sin(t)) + 0.5

# 5점 Gauss-Legendre [-1,1]
GL5_nodes = [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
GL5_weights = [0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]

# ------------------ 구간 변환 적용 Gauss-Legendre ------------------
def gauss_legendre(f, n, a, b):
    res = 0.0
    for i in range(5):
        t = 0.5*(b-a)*GL5_nodes[i] + 0.5*(a+b)
        w = GL5_weights[i]
        res += w * f(t, n)
    res *= 0.5*(b-a)
    return res

# ------------------ 적분 함수 ------------------
def integrand_a(t, n):
    return math.sqrt(PI**2 - t**2) * math.cos(n*t)

# def integrand_b(t, n):
#     return math.sqrt(PI**2 - t**2) * math.sin(n*t)

# ------------------ Fourier 계수 계산 ------------------
@lru_cache(maxsize=None)
def fourier_coefficient(n):
    if n == 0:  # a0
        return (PI**2)/4
    if n % 2 :  # b_m, n=1,3,5,... → b1,b2,...
        return 0
    else:  # a_m, n=2,4,6,... → a1,a2,...
        m = (n+1)//2
        integral = gauss_legendre(integrand_a, m, 0, PI)
        coef = integral * -2/PI if  m % 2 else integral * 2/PI
    return coef 

# TODO: optional. implement the semi circle wave function
def semi_circle_wave(t):
    return np.sqrt(PI**2 - (t - PI)**2)

def function(t):
    if signal_name == "square":
        return square_wave(t)
    elif signal_name == "semicircle":
        return semi_circle_wave(t)
    else:
        raise Exception("Unknown Signal")


def visualize():
    if not os.path.exists(f"{signal_name}_{N_Fourier}"):
        os.makedirs(f"{signal_name}_{N_Fourier}")

    frames = 100

    # x and y are for drawing the original function
    x = np.linspace(0, 2 * math.pi, 1000)
    y = np.zeros(1000, dtype = float)
    for i in range(1000):
        y[i] = function(x[i])

    for i in range(frames):
        figure, axes = plt.subplots()
        color=iter(cm.rainbow(np.linspace(0, 1, 2 * N_Fourier + 1)))

        time = 2 * math.pi * i / 100
        point_pos_array = np.zeros((2 * N_Fourier + 2, 2), dtype = float)
        radius_array = np.zeros((2 * N_Fourier + 1), dtype = float)

        point_pos_array[0, :] = [0, 0]
        radius_array[0] = fourier_coefficient(0)
        point_pos_array[1, :] = [0, radius_array[0]]

        circle = patches.Circle(point_pos_array[0], radius_array[0], fill = False, color = next(color))
        axes.add_artist(circle)

        f_t = function(time)
        for j in range(N_Fourier):
            # calculate circle for a_{n}
            radius_array[2 * j + 1] = fourier_coefficient(2 * j + 1)
            point_pos_array[2 * j + 2] = [point_pos_array[2 * j + 1][0] + radius_array[2 * j + 1] * math.cos((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 1][1] + radius_array[2 * j + 1] * math.sin((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 1], radius_array[2 * j + 1], fill = False, color = next(color))
            axes.add_artist(circle)
            
            # calculate circle for b_{n}
            radius_array[2 * j + 2] = fourier_coefficient(2 * j + 2)
            point_pos_array[2 * j + 3] = [point_pos_array[2 * j + 2][0] + radius_array[2 * j + 2] * math.sin((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 2][1] + radius_array[2 * j + 2] * math.cos((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 2], radius_array[2 * j + 2], fill = False, color = next(color))
            axes.add_artist(circle)
            
        plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], 'o-')
        plt.plot(x, y, '-')
        plt.plot([time, point_pos_array[-1][0]], [f_t, point_pos_array[-1][1]], '-', color = 'r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(f"{signal_name}_{N_Fourier}", "{}.png".format(i)))
        # plt.show()
        plt.close()
        
    images = []
    for i in range(frames):
        images.append(imageio.imread(os.path.join(f"{signal_name}_{N_Fourier}", "{}.png".format(i))))
    imageio.mimsave(f"{signal_name}_{N_Fourier}.mp4", images)


if __name__ == "__main__":
    visualize()