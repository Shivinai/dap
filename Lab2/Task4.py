import numpy as np
from scipy.io import loadmat
from scipy.io import whosmat
from scipy.signal import convolve
import matplotlib.pyplot as plt
from Task2 import circ_conv
import time

def quick_conv(x, h):
    N1 = len(x)
    N2 = len(h)
    N_out = N1 + N2 - 1

    x_padded = np.concatenate([x, np.zeros(N_out - N1)])
    h_padded = np.concatenate([h, np.zeros(N_out - N2)])

    y = np.zeros(N_out)

    y = np.fft.ifft(np.fft.fft(x_padded) * np.fft.fft(h_padded))

    return y

if __name__ == "__main__":
    print(whosmat('var_5.mat'))

    h = loadmat('var_5.mat')['h'].squeeze()
    x = loadmat('var_5.mat')['x'].squeeze()

    start_time_qc = time.perf_counter()
    y_qc = quick_conv(x, h)
    end_time_qc = time.perf_counter()
    time_qc = end_time_qc - start_time_qc

    start_time_circ = time.perf_counter()
    y_circ = circ_conv(x, h)
    end_time_circ = time.perf_counter()
    time_circ = end_time_circ - start_time_circ

    print(f"Quick circular convolution time: {time_qc:.6f} s")
    print(f"Not quick circular convolution time: {time_circ:.6f} s")

    n = np.arange(len(y_qc))

    plt.figure(figsize=(12,6))
    plt.plot(n, y_qc)
    plt.title("Quick convolution")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.show()