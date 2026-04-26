import numpy as np
from scipy.io import wavfile
from Task5 import write_wav
from scipy.signal import spectrogram
from scipy.signal import convolve
import matplotlib.pyplot as plt

def reverb(x, h, f_size):
    x = x[:f_size * (len(x) // f_size)]
    h = h[:f_size * (len(h) // f_size)]

    N_frames_x = len(x) // f_size
    N_frames_h = len(h) // f_size

    s_size = 2 * f_size - 1
    N_out = len(x) + len(h) - 1
    c_size = f_size + len(h) - 1 
    
    y_mtx = np.zeros((N_frames_x, N_out)) 
    mem = np.zeros(f_size - 1)          

    for j in range(N_frames_x):
        j_beg = j * f_size
        j_end = j_beg + f_size

        buf_x = np.concatenate((x[j_beg:j_end], np.zeros(f_size - 1)))
        
        y_tmp = np.zeros(c_size) 

        for i in range(N_frames_h):
            i_beg = i * f_size
            i_end = i_beg + f_size

            buf_h = np.concatenate((h[i_beg:i_end], np.zeros(f_size - 1)))

            y_k = np.fft.ifft(np.fft.fft(buf_x) * np.fft.fft(buf_h)).real

            buf = y_k[:f_size].copy()   
            buf[:f_size - 1] += mem      
            
            mem = y_k[f_size:]        
            
            y_tmp[i_beg:i_end] = buf
        
        y_tmp[N_frames_h * f_size : ] = mem

        mem.fill(0)

        col_beg = j * f_size
        col_end = col_beg + c_size
        y_mtx[j, col_beg:col_end] = y_tmp

    y = np.sum(y_mtx, axis=0)
    
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
   

    return y

if __name__ == "__main__":
    frame_size = 512
    sr_h, h_stereo = wavfile.read('fcl.wav')
    h = np.mean(h_stereo, axis=1).astype(h_stereo.dtype)

    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h))

    sr_x, x = wavfile.read("dc.wav")

    y = reverb(x, h, frame_size)
    write_wav('dc+reverb.wav', y, sr_x)
    y_conv = convolve(x, h)

    f1,t1,S1 = spectrogram(y, sr_x)
    f2,t2,S2 = spectrogram(y_conv, sr_x)

    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.pcolormesh(t1,f1,10*np.log10(S1 + 1e-10), shading='gouraud')
    plt.title("Section convolution")
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.subplot(2,1,2)
    plt.pcolormesh(t2,f2,10*np.log10(S2 + 1e-10), shading='gouraud')
    plt.title("MATLAB convolution")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()