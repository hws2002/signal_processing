import librosa
import numpy as np
import argparse

OUTPUT_FILE_NAME = 'output'

def key_tone_recognition(audio_array):
    '''
        请大家实现这一部分代码
    '''
    fs = audio_array[1] # 48000 Hz
    signal = np.array(audio_array[0], dtype=np.float32)

    frame_duration = 1.0 / 64.0
    frame_len = int(round(fs * frame_duration)) # 750
    if frame_len <= 0:
        with open(OUTPUT_FILE_NAME + '.txt', 'w', encoding='utf-8') as f:
            f.write('')
        return

    num_frames = len(signal) // frame_len
    if num_frames <= 0:
        with open(OUTPUT_FILE_NAME + '.txt', 'w', encoding='utf-8') as f:
            f.write('')
        return

    trimmed = signal[:num_frames * frame_len]
    frames = trimmed.reshape(num_frames, frame_len)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    max_rms = float(np.max(rms)) if rms.size > 0 else 0.0

    labels = []

    if max_rms <= 0.0:
        labels = ['-1'] * num_frames
    else:
        silence_threshold = 0.1 * max_rms

        low_freqs = np.array([697, 770, 852, 941], dtype=np.float32)
        high_freqs = np.array([1209, 1336, 1477], dtype=np.float32)

        dtmf_map = {
            (697, 1209): '1', (697, 1336): '2', (697, 1477): '3',
            (770, 1209): '4', (770, 1336): '5', (770, 1477): '6',
            (852, 1209): '7', (852, 1336): '8', (852, 1477): '9',
            (941, 1209): '*', (941, 1336): '0', (941, 1477): '#',
        }

        N = frame_len
        window = np.hamming(N)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)

        low_indices = [int(np.argmin(np.abs(freqs - f))) for f in low_freqs]
        high_indices = [int(np.argmin(np.abs(freqs - f))) for f in high_freqs]

        for i in range(num_frames):
            if rms[i] < silence_threshold:
                labels.append('-1')
                continue

            frame = frames[i] * window
            spectrum = np.fft.rfft(frame)
            mag = np.abs(spectrum)

            low_amps = [mag[idx] for idx in low_indices]
            high_amps = [mag[idx] for idx in high_indices]

            best_low = int(np.argmax(low_amps))
            best_high = int(np.argmax(high_amps))

            low_f = int(low_freqs[best_low])
            high_f = int(high_freqs[best_high])

            key = dtmf_map.get((low_f, high_f), '-1')
            labels.append(str(key))

    output_path = OUTPUT_FILE_NAME + '.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type = str, default='./test.wav', help = 'test file name', required = True)
    args = parser.parse_args()
    input_audio_array = librosa.load(args.audio_file, sr = 48000, dtype = np.float32) # audio file is numpy float array
    key_tone_recognition(input_audio_array)