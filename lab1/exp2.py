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

parser.add_argument('--N_Fourier', type=int, default=128, help='Number of Fourier Series')
args = parser.parse_args()

N_Fourier = args.N_Fourier


signal_name = "semicircle"


# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am.
 

NUM_POINTS = 125
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

# Fourier series approximation (predicted value)
def fourier_approximation(t, n_terms):
    """
    Calculate approximation at t using n_terms Fourier coefficients
    """
    result = fourier_coefficient(0)  # a0
    for j in range(n_terms):
        # b_m term
        result += fourier_coefficient(2 * j + 1) * math.sin((j + 1) * t)
        # a_m term
        result += fourier_coefficient(2 * j + 2) * math.cos((j + 1) * t)
    return result

# Compare actual vs predicted values
def compare_values():
    """
    Compare actual and predicted values at multiple time points and calculate errors
    """
    test_points = np.linspace(0, 2 * PI, 100)
    actual_values = np.array([function(t) for t in test_points])
    predicted_values = np.array([fourier_approximation(t, N_Fourier) for t in test_points])

    # Calculate errors
    mse = np.mean((actual_values - predicted_values) ** 2)
    mae = np.mean(np.abs(actual_values - predicted_values))
    max_error = np.max(np.abs(actual_values - predicted_values))

    print(f"=== Actual vs Predicted Comparison (N_Fourier={N_Fourier}) ===")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Max Error: {max_error:.6f}")
    print("\nSample Comparison (10 points):")
    print(f"{'Time(t)':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 48)
    for i in range(0, 100, 10):
        t = test_points[i]
        actual = actual_values[i]
        predicted = predicted_values[i]
        error = abs(actual - predicted)
        print(f"{t:<12.4f} {actual:<12.6f} {predicted:<12.6f} {error:<12.6f}")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(test_points, actual_values, 'b-', label='Actual', linewidth=2)
    plt.plot(test_points, predicted_values, 'r--', label='Predicted (Fourier)', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.title(f'Actual vs Predicted (N={N_Fourier})')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_points, actual_values - predicted_values, 'g-', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Error')
    plt.title('Error (Actual - Predicted)')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    comparison_path = os.path.join("media", f'{signal_name}_{N_Fourier}_comparison.png')
    plt.savefig(comparison_path, dpi=150)
    print(f"\nComparison plot saved: {comparison_path}")
    plt.show()


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
    compare_values()