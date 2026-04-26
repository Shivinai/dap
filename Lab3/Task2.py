import numpy as np
from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

src_dir = Path(__file__).parent
file_path = src_dir / "test_signal.wav"
sr, x = wavfile.read(file_path)

f_size = 128
h_size = int(0.5 * f_size)
H = f_size - h_size
N_bins = (f_size // 2) + 1
N_frames = ((len(x) - f_size) // H) + 1

print("Bins: ", N_bins)
print("Frames: ", N_frames)

w_analysis = np.hanning(f_size)
w_synthesis = np.bartlett(f_size)

y = np.zeros(len(x))

for i in range(N_frames):
    i_beg = i * H
    i_end = i_beg + f_size

    buf1 = x[i_beg:i_end]

    frame = buf1 * w_analysis

    S = np.fft.rfft(frame, n=f_size)

    magnitude = np.abs(S)
    phase = np.random.uniform(0, 2 * np.pi, len(S))

    S_processed = magnitude * np.exp(1j * phase)

    iframe = np.fft.irfft(S_processed, n=f_size)

    buf2 = iframe * w_synthesis

    y[i_beg:i_end] += buf2

y_rescaled = np.int16(np.clip(y, -32768, 32767))
output_file = src_dir / "t2_o.wav"
wavfile.write(output_file, sr, y_rescaled)

print("Files written to disk")

font_dir = ['/usr/share/fonts/TTF/']
for font_file in fm.findSystemFonts(fontpaths=font_dir):
    fm.fontManager.addfont(font_file)

font_label = {'fontname': 'Times New Roman', 'size': 12}
font_title = {'fontname': 'Times New Roman', 'size': 14, 'weight': 'bold'}

time_axis = np.linspace(0, len(x) / sr, num=len(x))

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(time_axis, x, color='steelblue')
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
S1, f1, t1, im1 = plt.specgram(x, NFFT=f_size, Fs=sr, noverlap=h_size, mode='magnitude', scale='dB', vmin=-65, vmax=15, cmap='viridis')
plt.title('Source signal spectrogram', **font_title)
plt.xlabel('Time, S', **font_label)
plt.ylabel('Freq, Hz', **font_label)
plt.ylim(0, sr/2)
cbar1 = plt.colorbar(im1)
cbar1.set_label('Amp', **font_label)

plt.subplot(2, 2, 4)
S2, f2, t2, im2 = plt.specgram(y, NFFT=f_size, Fs=sr, noverlap=h_size, mode='magnitude', scale='dB', vmin=-65, vmax=15, cmap='viridis')
plt.title('Processed signal spectrogram', **font_title)
plt.xlabel('Time, S', **font_label)
plt.ylabel('Freq, Hz', **font_label)
plt.ylim(0, sr/2)
cbar2 = plt.colorbar(im2)
cbar2.set_label('Amp', **font_label)

plt.tight_layout()
plt.show()