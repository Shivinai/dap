import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.io import wavfile
from pathlib import Path

sr = 44100
T = 5
t = np.arange(0, T, 1/sr)

f_start1, f1, amp1 = 2000, 2, 500
frc1 = amp1 * np.sin(2 * np.pi * f1 * t) + f_start1
phase1 = np.cumsum(frc1 / sr)
y1 = np.sin(2 * np.pi * phase1)

f_start2, f2, amp2 = 4000, 1, 800
frc2 = amp2 * np.sin(2 * np.pi * f2 * t) + f_start2
phase2 = np.cumsum(frc2 / sr)
y2 = np.sin(2 * np.pi * phase2)

f_start3, f3, amp3 = 7000, 3, 1000
frc3 = amp3 * np.sin(2 * np.pi * f3 * t) + f_start3
phase3 = np.cumsum(frc3 / sr)
y3 = np.sin(2 * np.pi * phase3)

y = y1 + y2 + y3

y_norm = y / np.max(np.abs(y))

y_rescaled = np.int16(y_norm * 32767)

src_dir = Path(__file__).parent
file_path = src_dir / "t1.1_o.wav"

wavfile.write(file_path, sr, y_rescaled)

print("Files written to disk")

font_dir = ['/usr/share/fonts/TTF/']
for font in fm.findSystemFonts(fontpaths=font_dir):
    fm.fontManager.addfont(font)

font = {'fontname': 'Times New Roman', 'size': 14}

f_size = 1024
h_size = 256

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