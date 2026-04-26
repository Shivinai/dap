import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from Task5 import write_wav

def crm(x, sr, f_start, f_end, step, deltat):
    n = len(x)
    y = np.zeros_like(x)
    
    phase = 0
    flag = 0
    f_ccur = f_start

    for i in range(0, n, deltat):
        block_size = min(deltat, n - i)

        phase_arr = phase + np.arange(block_size) * (2 * np.pi * f_ccur / sr)
        m_block = np.sin(phase_arr)

        y[i : i + block_size] = x[i : i + block_size] * m_block

        phase = (phase_arr[-1] + (2 * np.pi * f_ccur / sr)) % (2 * np.pi)

        if block_size == deltat:
            if f_ccur >= f_end:
                flag = 1
            elif f_ccur <= f_start:
                flag = 0
                
            if flag == 0:
                f_ccur += step
            elif flag == 1:
                f_ccur -= step
                
    return y

if __name__ == "__main__":
    f_start = 300
    f_end = 2000
    step = 30
    deltat = 512

    sr, x = wavfile.read('test_signal.wav')

    y = crm(x, sr, f_start, f_end, step, deltat)

    write_wav('test_signal_modulated.wav', y, sr)

    n = np.arange(len(x))

    f1,t1,S1 = spectrogram(x, sr)
    f2,t2,S2 = spectrogram(y, sr)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.plot(n, x)
    plt.title("Source signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    #plt.ylim(0, 4000)
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.pcolormesh(t1, f1, 10*np.log10(S1 + 1e-10), shading='gouraud')
    plt.title("Source signal spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 3)
    plt.plot(n, y)
    plt.title("Modulated signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    #plt.ylim(0, 4000)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.pcolormesh(t2, f2, 10*np.log10(S2 + 1e-10), shading='gouraud')
    plt.title("Modulated signal spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.show()