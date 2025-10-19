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

parser.add_argument('--N_Fourier', type=int, default=64, help='Number of Fourier Series')
args = parser.parse_args()

N_Fourier = args.N_Fourier


signal_name = "semicircle"


# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am.
 

NUM_POINTS = 300
GL_nodes, GL_weights = np.polynomial.legendre.leggauss(NUM_POINTS)

# ------------------ 구간 변환 적용 Gauss-Legendre ------------------
def gauss_legendre(f, n, a, b):
    res = 0.0
    for i in range(NUM_POINTS):
        t = 0.5*(b-a)*GL_nodes[i] + 0.5*(a+b)
        w = GL_weights[i]
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


def semi_circle_wave(t):
    return np.sqrt(PI**2 - (t - PI)**2)

def function(t):
    return semi_circle_wave(t)


FOLDER = f"media/{signal_name}_{N_Fourier}"
def visualize():
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    frames = 100

    # x and y are for drawing the original function
    x = np.linspace(0, 2 * math.pi, 2000)
    y = np.zeros(2000, dtype = float)
    for i in range(2000):
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
            # calculate circle for b_{n}
            radius_array[2 * j + 1] = fourier_coefficient(2 * j + 1)
            point_pos_array[2 * j + 2] = [point_pos_array[2 * j + 1][0] + radius_array[2 * j + 1] * math.cos((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 1][1] + radius_array[2 * j + 1] * math.sin((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 1], radius_array[2 * j + 1], fill = False, color = next(color))
            axes.add_artist(circle)
            
            # calculate circle for a_{n}
            radius_array[2 * j + 2] = fourier_coefficient(2 * j + 2)
            point_pos_array[2 * j + 3] = [point_pos_array[2 * j + 2][0] + radius_array[2 * j + 2] * math.sin((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 2][1] + radius_array[2 * j + 2] * math.cos((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 2], radius_array[2 * j + 2], fill = False, color = next(color))
            axes.add_artist(circle)
            
        plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], 'o-')
        plt.plot(x, y, '-')
        plt.plot([time, point_pos_array[-1][0]], [f_t, point_pos_array[-1][1]], '-', color = 'r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(FOLDER, "{}.png".format(i)))
        # plt.show()
        plt.close()
        
    images = []
    for i in range(frames):
        images.append(imageio.imread(os.path.join(FOLDER, "{}.png".format(i))))
    imageio.mimsave(os.path.join("media", f"{signal_name}_{N_Fourier}.mp4"), images)


if __name__ == "__main__":
    visualize()