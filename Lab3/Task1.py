import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.io import wavfile
from pathlib import Path

sr = 44100
T = 5
f_start = 12000
#f = np.random.randint(0, 200)
f = 10
amp = 6000

f_size = 1024
h_size = 256

print("Frequency: ", f)

t = np.arange(0, T, 1/sr)

frc = amp * np.sinc(2 * f * t) + f_start
phase = np.cumsum(frc / sr)

y = np.sin(2 * np.pi * phase)

font_dir = ['/usr/share/fonts/TTF/']
for font in fm.findSystemFonts(fontpaths=font_dir):
    fm.fontManager.addfont(font)

font = {'fontname': 'Times New Roman', 'size': 14}

plt.figure(figsize=(12, 7))

S1,f1,t1, im = plt.specgram(y, NFFT=f_size, Fs=sr, noverlap=h_size, mode='magnitude',  scale='dB', vmin=-65, vmax=15, cmap='viridis')

plt.xlabel('Time, S', **font)
plt.ylabel('Freq, Hz', **font)
plt.title('Spectrogram', **font)

plt.ylim(0, sr/2)

cbar = plt.colorbar(im)
cbar.set_label('Amp', **font)

plt.tight_layout()
plt.show()

src_dir = Path(__file__).parent
file_path = src_dir / "t1_o.wav"

y_rescaled = np.int16(y / np.max(np.abs(y)) * 32767)
wavfile.write(file_path, sr, y_rescaled)