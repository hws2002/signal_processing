import numpy as np
import matplotlib.pyplot as plt

def plot_aliasing_example():
    # 1. 설정값 (아까 든 예시 그대로)
    f_original = 7     # 원래 신호 주파수: 7Hz (범인)
    f_sampling = 10    # 샘플링 주파수: 10Hz (부족한 목격자)
    duration = 1.0     # 보여줄 시간: 1초
    
    # 에일리어싱 주파수 계산 |10 - 7| = 3Hz
    f_alias = abs(f_sampling - f_original) 

    # 2. 시간축 생성
    # (1) 아날로그 신호처럼 보이게 아주 촘촘하게 그림 (Continuous)
    t_fine = np.linspace(0, duration, 1000)
    
    # (2) 실제 샘플링되는 시점 (Discrete)
    # 0초부터 0.1초(1/10) 간격으로 점을 찍음
    t_samples = np.arange(0, duration + 1/f_sampling, 1/f_sampling)

    # 3. 신호 만들기
    # 원래 신호 (7Hz)
    y_original = np.cos(2 * np.pi * f_original * t_fine + np.pi/6)
    
    # 샘플링된 데이터 (원래 신호에서 점만 쏙쏙 뽑음)
    y_samples = np.cos(2 * np.pi * f_original * t_samples + np.pi/6)
    
    # 에일리어싱된 가짜 신호 (3Hz)
    y_alias = np.cos(2 * np.pi * f_alias * t_fine - np.pi/6)

    # 4. 그래프 그리기
    plt.figure(figsize=(12, 6))

    # (1) 원래 신호 (회색 점선) - 빠르게 진동함
    plt.plot(t_fine, y_original, 'gray', linestyle='--', alpha=0.5, label=f'Original Signal ({f_original} Hz)')

    # (2) 에일리어싱 신호 (빨간 실선) - 느리게 진동함
    plt.plot(t_fine, y_alias, 'r', linewidth=2, label=f'Aliased Signal ({f_alias} Hz)')

    # (3) 샘플링 포인트 (파란 점)
    plt.scatter(t_samples, y_samples, color='blue', s=100, zorder=5, label='Samples (10 Hz)')
    
    # 시각적 보조선 (Stem plot)
    plt.stem(t_samples, y_samples, linefmt='b:', markerfmt='bo', basefmt=" ")

    # 그래프 꾸미기
    plt.title(f'Aliasing Effect: {f_original}Hz sampled at {f_sampling}Hz appears as {f_alias}Hz')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.show()

# 함수 실행
plot_aliasing_example()