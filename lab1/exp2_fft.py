import numpy as np
import math
import cmath  # 복소수 연산을 위한 cmath 라이브러리
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm
import argparse
from functools import lru_cache

PI = math.pi

# --- 1. 기본 설정 ---
parser = argparse.ArgumentParser()
parser.add_argument('--N_Fourier', type=int, default=128, help='Number of Fourier Series')
args = parser.parse_args()
N_Fourier = args.N_Fourier
signal_name = "semicircle"
FOLDER = f"media/{signal_name}_{N_Fourier}_manual_fft"


# --- 2. 신호 함수 정의 ---
def semi_circle_wave(t):
    return np.sqrt(PI**2 - (t - PI)**2)

def function(t):
    return semi_circle_wave(t)


# --- 3. FFT 직접 구현 (라이브러리 미사용) ---
def fft_manual(x):
    """Cooley-Tukey Radix-2 DIT FFT 알고리즘을 순수 Python으로 구현합니다."""
    N = len(x)
    if N <= 1: return x
    
    # 1단계: 비트 역순 재정렬
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j: x[i], x[j] = x[j], x[i]

    # 2단계: 버터플라이 연산
    block_size = 2
    while block_size <= N:
        half_block = block_size // 2
        angle = -2 * PI / block_size
        twiddle_base = cmath.exp(complex(0, angle))
        for i in range(0, N, block_size):
            twiddle = 1.0
            for j in range(half_block):
                k = i + j
                temp = twiddle * x[k + half_block]
                x[k + half_block] = x[k] - temp
                x[k] = x[k] + temp
                twiddle *= twiddle_base
        block_size <<= 1
    return x


# --- 4. 직접 구현한 FFT를 이용해 모든 계수를 미리 계산 ---
@lru_cache(maxsize=None)
def calculate_all_coeffs_manual(num_coeffs, N_samples=2048):
    """직접 구현한 FFT를 사용하여 푸리에 계수를 계산합니다."""
    if not (N_samples and (N_samples & (N_samples - 1) == 0)):
        raise ValueError(f"FFT 샘플 개수는 2의 거듭제곱이어야 합니다. 현재: {N_samples}")
    
    t = np.linspace(0, 2 * PI, N_samples, endpoint=False)
    y = semi_circle_wave(t).tolist()
    fft_result = fft_manual(y)
    
    a0 = fft_result[0].real / N_samples
    m_indices = np.arange(1, num_coeffs + 1)
    a_m = 2 * np.array([fft_result[m].real for m in m_indices]) / N_samples
    b_m = -2 * np.array([fft_result[m].imag for m in m_indices]) / N_samples
    
    return a0, a_m, b_m

# 프로그램 시작 시 모든 계수를 한 번만 계산하여 전역 변수에 저장
print("수동 구현 FFT로 푸리에 계수를 계산 중입니다...")
A0_FFT, A_M_FFT, B_M_FFT = calculate_all_coeffs_manual(N_Fourier)
print("계산 완료.")


# --- 5. visualize() 함수가 호출할 계수 반환 헬퍼 함수 ---
def fourier_coefficient(n):
    """
    미리 계산된 FFT 결과에서 n에 맞는 계수를 찾아 반환합니다.
    n=0: a0, n=1: b1, n=2: a1, n=3: b2, n=4: a2 ...
    """
    if n == 0:
        return A0_FFT
    if n % 2:  # n이 홀수이면 b_m 계수
        m = (n + 1) // 2
        return B_M_FFT[m - 1] # m은 1부터 시작하므로 인덱스는 m-1
    else:  # n이 짝수이면 a_m 계수
        m = n // 2
        return A_M_FFT[m - 1]


# --- 6. 시각화 함수 ---
def visualize():
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    frames = 100
    x = np.linspace(0, 2 * math.pi, 2000)
    y = function(x)

    for i in range(frames):
        figure, axes = plt.subplots()
        color=iter(cm.rainbow(np.linspace(0, 1, 2 * N_Fourier + 1)))

        time = 2 * math.pi * i / frames
        point_pos_array = np.zeros((2 * N_Fourier + 2, 2), dtype = float)
        radius_array = np.zeros((2 * N_Fourier + 1), dtype = float)

        # a0 (DC offset) 처리
        point_pos_array[0, :] = [0, 0]
        radius_array[0] = fourier_coefficient(0)
        point_pos_array[1, :] = [0, radius_array[0]]
        
        # a0에 해당하는 원은 없으므로 색상만 하나 소모
        next(color)

        f_t = function(time)
        for j in range(N_Fourier):
            m = j + 1 # 주파수 m = 1, 2, 3...
            
            # b_m (n = 2m - 1)에 대한 원 계산
            # 원본 코드에서는 n=2j+1 이 b_m에 해당
            radius_array[2 * j + 1] = fourier_coefficient(2 * m - 1)
            point_pos_array[2 * j + 2] = [point_pos_array[2 * j + 1][0] + radius_array[2 * j + 1] * math.cos(m * time),
                                          point_pos_array[2 * j + 1][1] + radius_array[2 * j + 1] * math.sin(m * time)]
            circle = patches.Circle(point_pos_array[2 * j + 1], abs(radius_array[2 * j + 1]), fill=False, color=next(color))
            axes.add_artist(circle)
            
            # a_m (n = 2m)에 대한 원 계산
            # 원본 코드에서는 n=2j+2 가 a_m에 해당
            radius_array[2 * j + 2] = fourier_coefficient(2 * m)
            point_pos_array[2 * j + 3] = [point_pos_array[2 * j + 2][0] + radius_array[2 * j + 2] * math.sin(m * time),
                                          point_pos_array[2 * j + 2][1] + radius_array[2 * j + 2] * math.cos(m * time)]
            circle = patches.Circle(point_pos_array[2 * j + 2], abs(radius_array[2 * j + 2]), fill=False, color=next(color))
            axes.add_artist(circle)
        
        # 그리기
        plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], 'o-', markersize=2)
        plt.plot(x, y, '-')
        plt.plot([time, point_pos_array[-1][0]], [f_t, point_pos_array[-1][1]], '-', color='r')
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-5, 11)
        plt.ylim(-2, 5)
        
        plt.savefig(os.path.join(FOLDER, f"{i:03d}.png"))
        plt.close()
        print(f"Frame {i+1}/{frames} rendered.", end='\r')
    
    # 영상 파일 생성
    print("\nCreating MP4 file...")
    images = [imageio.imread(os.path.join(FOLDER, f"{i:03d}.png")) for i in range(frames)]
    imageio.mimsave(os.path.join(FOLDER, f"{signal_name}_{N_Fourier}.mp4"), images, fps=25)
    print("MP4 file saved successfully.")


if __name__ == "__main__":
    visualize()