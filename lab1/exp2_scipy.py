import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os
import imageio
from matplotlib.pyplot import cm
import argparse
from functools import lru_cache
from scipy.integrate import quad  # SciPy의 적분 함수 import

PI = math.pi

# --- 1. 기본 설정 및 인자 파서 ---
parser = argparse.ArgumentParser()
parser.add_argument('--N_Fourier', type=int, default=16, help='Number of Fourier Series')
args = parser.parse_args()

N_Fourier = args.N_Fourier
signal_name = "semicircle"


# --- 2. 푸리에 계수 계산 (Scipy 버전) ---

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

# --- 3. 신호 함수 정의 ---
def semi_circle_wave(t):
    # t가 배열일 경우를 대비하여 np.sqrt 사용
    return np.sqrt(PI**2 - (t - PI)**2)

# --- 4. 시각화 함수 ---
def visualize():
    folder_name = f"{signal_name}_{N_Fourier}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    frames = 120  # 프레임 수를 늘려 더 부드러운 영상 생성

    # 원본 함수 그리기 위한 데이터
    x_original = np.linspace(0, 2 * PI, 1000)
    y_original = semi_circle_wave(x_original)

    # 근사 함수를 추적하기 위한 데이터
    y_approx_trace = []
    x_approx_trace = np.linspace(0, 2 * PI, frames, endpoint=False)

    for i in range(frames):
        figure, axes = plt.subplots(figsize=(8, 8))
        color = iter(cm.viridis(np.linspace(0, 1, N_Fourier + 1)))

        time = 2 * PI * i / frames
        
        # 시작점: a0 (DC offset)
        center_pos = np.array([0, fourier_coefficient(0)])
        
        # 회전 벡터(Epicycle)들의 중심점을 저장할 리스트
        epicycle_centers = [center_pos]
        
        # a_m 항들에 대한 회전 벡터 계산 (b_m 항은 0이므로 생략)
        for m in range(1, N_Fourier + 1):
            radius = fourier_coefficient(2 * m) # a_m 계수
            prev_center = epicycle_centers[-1]
            
            # 표준 좌표계 (R*cos(wt), R*sin(wt))를 사용한 회전 벡터
            new_x = prev_center[0] + radius * math.cos(m * time)
            new_y = prev_center[1] + radius * math.sin(m * time)
            
            # 원 그리기
            circle = patches.Circle(prev_center, abs(radius), fill=False, color=next(color), ls='--')
            axes.add_artist(circle)
            
            epicycle_centers.append(np.array([new_x, new_y]))

        epicycle_centers = np.array(epicycle_centers)
        final_point = epicycle_centers[-1]
        
        # 근사된 y값 추적
        if len(y_approx_trace) < frames:
             y_approx_trace.append(final_point[1])

        # 화면에 그리기
        axes.plot(epicycle_centers[:, 0], epicycle_centers[:, 1], 'o-', color='gray', markersize=3, alpha=0.8)
        axes.plot(x_original, y_original, '-', color='blue', lw=2, label='Original Signal')
        
        # 근사 함수가 그려지는 경로
        if y_approx_trace:
            axes.plot(x_approx_trace[:len(y_approx_trace)], y_approx_trace, '-', color='red', lw=2, label=f'Approximation (N={N_Fourier})')
        
        # 회전 벡터의 끝에서 근사 그래프까지 수평선 연결
        axes.plot([final_point[0], x_approx_trace[i]], [final_point[1], final_point[1]], '--', color='black', lw=0.8)
        
        axes.set_title(f'Fourier Series Epicycles Visualization')
        axes.legend()
        axes.grid(True, linestyle=':', alpha=0.6)
        axes.set_aspect('equal', adjustable='box')
        axes.set_xlim(-5, 11)
        axes.set_ylim(-2, 5)

        plt.savefig(os.path.join(folder_name, f"{i:03d}.png"))
        plt.close()
        print(f"Frame {i+1}/{frames} rendered.", end='\r')

    # MP4 파일 생성
    print("\nCreating MP4 file...")
    images = []
    for i in range(frames):
        images.append(imageio.imread(os.path.join(folder_name, f"{i:03d}.png")))
    imageio.mimsave(f"{folder_name}.mp4", images, fps=30)
    print("MP4 file saved successfully.")


if __name__ == "__main__":
    visualize()