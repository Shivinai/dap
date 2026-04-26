import numpy as np
from scipy.io import loadmat
from scipy.io import whosmat
from scipy.signal import convolve
import matplotlib.pyplot as plt

def circ_conv(x, h):
    N1 = len(x)
    N2 = len(h)
    N_out = max(N1, N2)

    x_padded = np.concatenate([x, np.zeros(N_out - N1)])
    h_padded = np.concatenate([h, np.zeros(N_out - N2)])

    y = np.zeros(N_out)
    buf = np.zeros(N_out)
    buf = np.flip(h_padded)

    for i in range(N_out):
        tmp = 0
        buf = np.roll(buf, 1)

        for j in range(N_out):
            tmp += buf[j] * x_padded[j]
        
        y[i] = tmp

    return y

def matlab_reference_cyclic_convolve(x, h):
    n = max(len(x), len(h))
    
    X = np.fft.fft(x, n)
    H = np.fft.fft(h, n)
    
    return np.fft.ifft(X * H).real

if __name__ == "__main__":
    print(whosmat('var_5.mat'))

    h = loadmat('var_5.mat')['h'].squeeze()
    x = loadmat('var_5.mat')['x'].squeeze()

    y_circ = circ_conv(x, h)
    y_cconv = matlab_reference_cyclic_convolve(x, h)

    n1 = np.arange(len(y_circ))
    n2 = np.arange(len(x))
    n3 = np.arange(len(h))
    n4 = np.arange(len(y_cconv))

    plt.figure(figsize=(12,6))
    plt.subplot(2,2,1)
    plt.plot(n1, y_circ)
    plt.title("Circular convolution")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(n4, y_cconv)
    plt.title("Cconv")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(n2, x)
    plt.title("X signal")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(n3, h)
    plt.title("H signal")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

    error = y_circ - y_cconv
    mse = np.mean(error**2)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(y_circ, label="Circular convolution", alpha=0.7)
    plt.plot(y_cconv, label="Cconv()", linestyle='--', alpha=0.7)
    plt.title("Signal comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(error, color='red')
    plt.title(f"Absolute error (MSE = {mse:.2e})")
    plt.xlabel("n")
    plt.ylabel("Diff")
    plt.grid(True)

    plt.tight_layout()
    plt.show()