# 实验二报告

## 实现方法

### 1. 音频分帧

- 采样率：48000 Hz
- 帧长：1/64 秒 = 750 个采样点
- 将整个音频信号按帧长切分为若干帧

```python
frame_duration = 1.0 / 64.0
frame_len = int(round(fs * frame_duration))  # 750
num_frames = len(signal) // frame_len
frames = signal[:num_frames * frame_len].reshape(num_frames, frame_len)
```

### 2. 静默检测

计算每帧的均方根能量（RMS）：

$$
\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
$$

设定阈值为最大RMS的10%，低于阈值的帧判定为静默，输出 `-1`。

```python
rms = np.sqrt(np.mean(frames ** 2, axis=1))
silence_threshold = 0.1 * np.max(rms)
```

### 3. 按键识别

对非静默帧进行短时傅里叶变换（STFT）：

1. 对每帧加 Hamming 窗以减少频谱泄漏
2. 使用 `np.fft.rfft` 计算频谱
3. 在低频组和高频组的对应频率位置提取幅度
4. 分别找出低频组和高频组中幅度最大的频率
5. 根据频率对查表得到对应按键

```python
window = np.hamming(frame_len)
frame_windowed = frame * window
spectrum = np.fft.rfft(frame_windowed)
mag = np.abs(spectrum)

# 找到低频和高频中能量最大的频率
best_low = np.argmax([mag[idx] for idx in low_indices])
best_high = np.argmax([mag[idx] for idx in high_indices])

# 查表得到按键
key = dtmf_map.get((low_freqs[best_low], high_freqs[best_high]), '-1')
```


## 运行方式

```bash
python main.py --audio_file test.wav
```

运行后在同目录下生成 `output.txt`。
