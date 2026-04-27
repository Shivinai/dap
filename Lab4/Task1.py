import numpy as np
from scipy.io import wavfile
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def noise_signal(signal, target_snr):
    signal_float = signal.astype(np.float64)
    rms = np.sqrt(np.mean(signal_float**2))

    rms_noise = rms / (10**(target_snr/20))
    noise = np.random.normal(0, rms_noise, len(signal_float))

    signal_noised = signal_float + noise
    signal_noised = np.clip(signal_noised, -32768, 32767).astype(np.int16)

    return signal_noised

if __name__ == "__main__":
    src_dir = Path(__file__).parent
    file_path = src_dir / "05.wav"

    sr, x = wavfile.read(file_path)

    y1 = noise_signal(x, 5)
    y2 = noise_signal(x, 10)
    y3 = noise_signal(x, 20)

    wavfile.write(src_dir / "t1_o1.wav", sr, y1)
    wavfile.write(src_dir / "t1_o2.wav", sr, y2)
    wavfile.write(src_dir / "t1_o3.wav", sr, y3)
    print("Files written to disk")

    font_dir = ['/usr/share/fonts/TTF/']
    for font_file in fm.findSystemFonts(fontpaths=font_dir):
        fm.fontManager.addfont(font_file)

    font_label = {'fontname': 'Times New Roman', 'size': 12}
    font_title = {'fontname': 'Times New Roman', 'size': 14}

    spec_params = {
        'NFFT': 1024, 
        'Fs': sr, 
        'noverlap': 512, 
        'mode': 'magnitude', 
        'scale': 'dB', 
        'vmin': -65, 
        'vmax': 15, 
        'cmap': 'viridis'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    signals = [
        (x, 'Original Signal'),
        (y1, 'SNR = 5 dB'),
        (y2, 'SNR = 10 dB'),
        (y3, 'SNR = 20 dB')
    ]

    for ax, (sig, title) in zip(axes, signals):
        S, f, t, im = ax.specgram(sig, **spec_params)
        ax.set_xlabel('Time, s', **font_label)
        ax.set_ylabel('Freq, Hz', **font_label)
        ax.set_title(title, **font_title)
        ax.set_ylim(0, sr/2)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amp, dB', **font_label)

    plt.suptitle('Spectrograms', fontsize=14, fontname='Times New Roman')
    plt.tight_layout()
    plt.show()