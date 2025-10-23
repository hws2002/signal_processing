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

# TODO: 1. Change N_Fourier to 2, 4, 8, 16, 32, 64, 128, get visualization results with differnet number of Fourier Series
N_Fourier = args.N_Fourier

# TODO: optional, implement visualization for semi-circle
signal_name = "square"

# TODO: 2. Please implement the function that calculates the Nth fourier coefficient
# Note that n starts from 0
# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am. 

@lru_cache(maxsize=None)
def fourier_coefficient(n):
    if n == 0 :
        return 1/2
    if n % 2 : # bm
        m = (n+1)//2
        if m % 2 : # odd
            coef = 2/(m * PI)
        else : # even
            coef = 0
    else : # am
        coef = 0
    return coef

# TODO: 3. implement the signal function
@lru_cache(maxsize=None)
def square_wave(t):
    return 0.5 * np.sign(math.sin(t)) + 0.5

# TODO: optional. implement the semi circle wave function
def semi_circle_wave(t):
    pass

def function(t):
    if signal_name == "square":
        return square_wave(t)
    elif signal_name == "semicircle":
        return semi_circle_wave(t)
    else:
        raise Exception("Unknown Signal")

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
    plt.savefig(f'{signal_name}_{N_Fourier}_comparison.png', dpi=150)
    print(f"\nComparison plot saved: {signal_name}_{N_Fourier}_comparison.png")
    plt.show()


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
    compare_values()