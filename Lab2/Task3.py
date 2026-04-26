import numpy as np
from scipy.io import loadmat
from scipy.io import whosmat
from scipy.signal import convolve
import matplotlib.pyplot as plt
from Task1 import lin_conv

def lin_conv_via_circ(x, h):
    N1 = len(x)
    N2 = len(h)
    N_out = N1 + N2 - 1

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
    
if __name__ == "__main__":
    print(whosmat('var_5.mat'))

    h = loadmat('var_5.mat')['h'].squeeze()
    x = loadmat('var_5.mat')['x'].squeeze()

    y_lin_via_circ = lin_conv_via_circ(x, h)
    y_lin = lin_conv(x, h)

    n1 = np.arange(len(y_lin_via_circ))
    n2 = np.arange(len(y_lin))

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(n1, y_lin_via_circ)
    plt.title("Linear convolution via circular")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(n2, y_lin)
    plt.title("Linear convolution")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()
    plt.show()