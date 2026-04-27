import numpy as np
from scipy.io import wavfile
from scipy.fft import dct, idct  
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

src_dir = Path(__file__).parent
file_path = src_dir / "test_signal.wav"

sr, x = wavfile.read(file_path)

SNR_dB = 80
RMS_x = np.sqrt(np.mean(x.astype(np.float64)**2))
RMS_noise = np.sqrt((RMS_x**2) / (10**(SNR_dB/10)))

noise = np.random.normal(0, RMS_noise, len(x))
x_noisy = x + noise

x_noisy = np.clip(x_noisy, -32768, 32767)

f_size = 128
h_size = int(0.5 * f_size)
H = f_size - h_size
N_frames = ((len(x_noisy) - f_size) // H) + 1

w_analysis = np.hanning(f_size)
w_synthesis = np.bartlett(f_size)

y = np.zeros(len(x_noisy))

for i in range(N_frames):
    i_beg = i * H
    i_end = i_beg + f_size

    buf1 = x_noisy[i_beg:i_end]

    frame = buf1 * w_analysis

    X = dct(frame, type=2, norm='ortho')

    coeffs = X[f_size // 2:]
    noise_median = np.median(np.abs(coeffs))
    
    sigma = noise_median / 0.6745 if noise_median > 0 else 1e-6
    
    threshold = sigma * np.sqrt(2 * np.log(f_size)) * 1.5 

    X_processed = np.where(np.abs(X) > threshold, X, 0)

    iframe = idct(X_processed, type=2, norm='ortho')

    buf2 = iframe * w_synthesis
    y[i_beg:i_end] += buf2

y = y * 1.5 
y_rescaled = np.int16(np.clip(y, -32768, 32767))

output_file = src_dir / "t3_o.wav"
wavfile.write(output_file, sr, y_rescaled)
print("Files written to disk")

font_dir = ['/usr/share/fonts/TTF/']
for font_file in fm.findSystemFonts(fontpaths=font_dir):
    fm.fontManager.addfont(font_file)

font_label = {'fontname': 'Times New Roman', 'size': 12}
font_title = {'fontname': 'Times New Roman', 'size': 14, 'weight': 'bold'}

time_axis = np.linspace(0, len(x_noisy) / sr, num=len(x_noisy))

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(time_axis, x_noisy, color='steelblue')
plt.title('Source signal', **font_title)
plt.xlabel('Time, S', **font_label)
plt.ylabel('Amp', **font_label)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(time_axis, y, color='rebeccapurple')
plt.title('Processed signal', **font_title)
plt.xlabel('Time, S', **font_label)
plt.ylabel('Amp', **font_label)
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
S1, f1, t1, im1 = plt.specgram(x_noisy, NFFT=f_size, Fs=sr, noverlap=h_size, mode='magnitude', scale='dB', vmin=-65, vmax=15, cmap='viridis')
plt.title('Source signal spectrogram', **font_title)
plt.xlabel('Time, S', **font_label)
plt.ylabel('Freq, Hz', **font_label)
plt.ylim(0, sr/2)
cbar1 = plt.colorbar(im1)
cbar1.set_label('Amp, dB', **font_label)

plt.subplot(2, 2, 4)
S2, f2, t2, im2 = plt.specgram(y, NFFT=f_size, Fs=sr, noverlap=h_size, mode='magnitude', scale='dB', vmin=-65, vmax=15, cmap='viridis')
plt.title('Processed signal spectrogram', **font_title)
plt.xlabel('Time, S', **font_label)
plt.ylabel('Freq, Hz', **font_label)
plt.ylim(0, sr/2)
cbar2 = plt.colorbar(im2)
cbar2.set_label('Amp, dB', **font_label)

plt.tight_layout()
plt.show()