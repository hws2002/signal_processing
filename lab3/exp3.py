import numpy as np
import matplotlib.pyplot as plt

# 时长为1秒
t = 1
# 采样率为60hz
fs = 60
t_split = np.arange(0, t * fs)


# 1hz与25hz叠加的正弦信号
x_1hz = t_split * 1 * np.pi * 2 / fs
x_25hz = t_split * 25 * np.pi * 2 / fs
signal_sin_1hz = np.sin(x_1hz)
signal_sin_25hz = np.sin(x_25hz)

signal_sin = signal_sin_1hz + 0.25 * signal_sin_25hz


# TODO: 补全这部分代码
# 通带边缘频率为10Hz，
# 阻带边缘频率为22Hz，
# 阻带衰减为44dB，窗内项数为17的汉宁窗函数
# 构建低通滤波器
# 函数需要返回滤波后的信号
def filter_fir(input):
    N = 17 #窗内系数
    fc = 16 #理想滤波器的截止频率
    w_c = 2 * np.pi * fc / fs #理想滤波器的数字频率
    
    n = np.arange(-(N-1)//2, (N-1)//2 + 1)
    # h_ideal = np.where(n == 0, w_c / np.pi, np.sin(w_c * n) / (np.pi * n)) #理想滤波器的脉冲响应
    # avoid runtime error
    h_ideal = np.zeros_like(n, dtype=float)   
    mask = (n != 0) 
    h_ideal[mask] = np.sin(w_c * n[mask]) / (np.pi * n[mask])
    h_ideal[~mask] = w_c / np.pi
    window = hanning(N)
    # window = np.hanning(N)
    h_n = h_ideal * window

    # filtered_signal = np.convolve(input, h_n)
    filtered_signal = convolve(input, h_n)
    return filtered_signal[:len(input)]

def hanning(N : int):
    return 0.54 + 0.46*np.cos(2 * np.pi * np.arange(-(N-1)//2, (N-1)//2 + 1)/ (N-1))

def convolve(N1, N2):
    L = len(N1)
    M = len(N2)
    res = np.zeros(L + M - 1)
    for n in range(L + M - 1):
        for m in range(max(0, n-M+1), min(L-1,n)+1):
            res[n] += N1[m]*N2[n-m]
    return res

# TODO: 首先正向对信号滤波(此时输出信号有一定相移)
# 将输出信号反向，再次用该滤波器进行滤波
# 再将输出信号反向
# 函数需要返回零相位滤波后的信号
def filter_zero_phase(input):
    delay_filtered_signal = filter_fir(input)
    reversed_delay_filtered_signal = delay_filtered_signal[::-1]
    zerophase_filtered_signal = filter_fir(reversed_delay_filtered_signal)
    return zerophase_filtered_signal[::-1]

if __name__ == "__main__":
    delay_filtered_signal = filter_fir(signal_sin)
    zerophase_filtered_signal = filter_zero_phase(signal_sin)

    plt.plot(t_split, signal_sin, label = 'origin')
    plt.plot(t_split, delay_filtered_signal, label = 'fir')
    plt.plot(t_split, zerophase_filtered_signal, label = 'zero phase')
    plt.legend()
    plt.show()
