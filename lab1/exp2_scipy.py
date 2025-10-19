import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os
import imageio
from matplotlib.pyplot import cm
from scipy.integrate import quad  # SciPy의 적분 함수 import


import argparse
from functools import lru_cache
PI = math.pi

parser = argparse.ArgumentParser()

parser.add_argument('--N_Fourier', type=int, default=32, help='Number of Fourier Series')
args = parser.parse_args()

N_Fourier = args.N_Fourier

signal_name = "semicircle"

# Note that n starts from 0
# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am.
 
@lru_cache(maxsize=None)
def square_wave(t):
    return 0.5 * np.sign(math.sin(t)) + 0.5


# quad 함수에 전달할 적분 대상 함수
# t_prime은 t' = t-pi 를 의미합니다.
def integrand_a_scipy(t_prime, m):
    return math.sqrt(PI**2 - t_prime**2) * math.cos(m * t_prime)

@lru_cache(maxsize=None)
def fourier_coefficient(n):
    """SciPy의 quad 함수를 사용하여 푸리에 계수를 계산합니다."""
    if n == 0:  # a0 (DC 성분)
        return (PI**2) / 4
    
    if n % 2:  # b_m 계수 (홀수 n) -> 이 신호에서는 0
        return 0
        
    else:  # a_m 계수 (짝수 n)
        m = n // 2  # 주파수 (a_1, a_2, ...)
        
        # quad를 이용한 수치 적분: 짝함수이므로 [0, PI] 구간을 적분
        # quad(함수, 시작, 끝, 추가 인자)는 (결과, 오차) 튜플을 반환합니다.
        integral_val, _ = quad(integrand_a_scipy, 0, PI, args=(m,))
        
        # a_m = (2*(-1)^m / PI) * [0, PI] 구간 적분값
        coef = (2 / PI) * integral_val * ((-1)**m)
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
        plt.savefig(os.path.join(f"{signal_name}_{N_Fourier}", "{}.png".format(i)))
        # plt.show()
        plt.close()
        
    images = []
    for i in range(frames):
        images.append(imageio.imread(os.path.join(f"{signal_name}_{N_Fourier}", "{}.png".format(i))))
    imageio.mimsave(f"{signal_name}_{N_Fourier}.mp4", images)


if __name__ == "__main__":
    visualize()