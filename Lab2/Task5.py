import numpy as np
from scipy.io import wavfile
from scipy.io import loadmat
from scipy.io import whosmat
import matplotlib.pyplot as plt
from Task4 import quick_conv
from scipy.signal import spectrogram
from scipy.signal import convolve

def quick_conv(x, h_fft, S):
    X = np.fft.fft(x, n=S)
    return np.real(np.fft.ifft(X * h_fft))

def section_convolution(x, h, L):
    N1 = len(x)
    M = len(h)
    
    if L < M:
        L = M 
        
    S = L + M - 1
    N_frames = int(np.ceil(N1 / L))
    N_out = N1 + M - 1
    
    y = np.zeros(N_out)
    mem = np.zeros(M - 1)
    
    H_fft = np.fft.fft(h, n=S)

    for i in range(N_frames):
        i_beg = i * L
        i_end = min(i_beg + L, N1) 
        
        x_chunk = x[i_beg:i_end]
        x_k = np.zeros(S)
        x_k[:len(x_chunk)] = x_chunk
        
        y_k = quick_conv(x_k, H_fft, S)
        
        y_k[:M-1] += mem

        mem = y_k[L:L+M-1]

        write_len = min(L, N_out - i_beg)
        y[i_beg : i_beg + write_len] = y_k[:write_len]

    y[N_frames * L : N_frames * L + len(mem)] += mem

    return y

def write_wav(filename, signal, sr):
    real_signal = np.real(signal)

    max_val = np.max(np.abs(real_signal))

    if max_val > 0:
        real_signal = real_signal / max_val

    signal_int16 = (real_signal * 32767).astype(np.int16)
    wavfile.write(filename, sr, signal_int16)
    print(f"Файл {filename} записан")

if __name__ == "__main__":
    sr, x_source = wavfile.read('test_signal.wav')
    L = 2048

    sect = len(x_source) // L
    new_len = sect * L

    x = x_source[:new_len]

    print(f"Source length: {len(x_source)}")
    print(f"Fitting L sections: {sect}")
    print(f"Trimmed lenght: {len(x)}")

    print(whosmat('filter_5.mat'))

    h = loadmat('filter_5.mat')['h'].squeeze()

    y_sc = section_convolution(x, h, L)
    y_conv = convolve(x, h)

    write_wav('result.wav', y_sc, sr)

    f1,t1,S1 = spectrogram(x, sr)
    f2,t2,S2 = spectrogram(y_sc, sr)
    f3,t3,S3 = spectrogram(y_conv, sr)

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.pcolormesh(t1,f1,10*np.log10(S1 + 1e-10), shading='gouraud')
    plt.title("Source signal")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.subplot(2,2,3)
    plt.pcolormesh(t2,f2,10*np.log10(S2 + 1e-10), shading='gouraud')
    plt.title("Section convolution")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.subplot(2,2,4)
    plt.pcolormesh(t3,f3,10*np.log10(S3 + 1e-10), shading='gouraud')
    plt.title("MATLAB convolution")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

