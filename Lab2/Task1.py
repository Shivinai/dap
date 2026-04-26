import numpy as np
from scipy.io import loadmat
from scipy.io import whosmat
from scipy.signal import convolve
import matplotlib.pyplot as plt

def lin_conv(x, h):
    x = np.asarray(x)
    h = np.asarray(h)
    
    N1 = len(x)
    N2 = len(h)
    N_out = N1 + N2 - 1

    x_padded = np.concatenate([x, np.zeros(N_out - N1)])
    h_padded = np.concatenate([h, np.zeros(N_out - N2)])

    y = np.zeros(N_out)
    buf = np.zeros(N_out)

    for i in range(N_out):
        buf[1:] = buf[:-1]
        buf[0] = h_padded[i]

        tmp = 0
        for j in range(N_out):
            tmp += buf[j] * x_padded[j]

        y[i] = tmp

    return y

if __name__ == "__main__":
    print(whosmat('var_5.mat'))

    h = loadmat('var_5.mat')['h'].squeeze()
    x = loadmat('var_5.mat')['x'].squeeze()

    y_lin = lin_conv(x, h)
    y_conv = convolve(x, h)

    n1 = np.arange(len(y_lin))
    n2 = np.arange(len(y_conv))

    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(n1, y_lin)
    plt.title("Linear convolution")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(n2, y_conv)
    plt.title("Convolve()")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

    error = y_lin - y_conv
    mse = np.mean(error**2)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(y_lin, label="Linear convolution", alpha=0.7)
    plt.plot(y_conv, label='Convolve()', linestyle='--', alpha=0.7)
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