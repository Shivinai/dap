import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

#vibecoded

def nextpow2(x):
    return int(np.ceil(np.log2(max(x, 1))))

def add_gaussian_noise_simple(signal, noise_std=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_std, signal.shape)
    return np.clip(signal + noise, -1.0, 1.0)

def processfunc(frame, frame_size, counter, noise_acc, Pdd, 
                N_tres=1.1, alpha=5.1, beta=0.002, gamma=0.9):
    nfft = 2 * (2 ** nextpow2(frame_size))
    
    if counter < 5:
        spectrum = np.fft.fft(frame, n=nfft)
        noise_acc += np.abs(spectrum)
        return frame * 1, (noise_acc, Pdd, counter + 1)

    if counter == 5:
        noise_acc = noise_acc / 5.0
        Pdd = noise_acc ** 2

    x_spectrum = np.fft.fft(frame, n=nfft)
    Pxx = np.abs(x_spectrum) ** 2
    Px = np.sum(Pxx)
    Pd = np.sum(Pdd)

    if 10 * np.log10(Px / Pd) < N_tres:
        Pdd = gamma * Pdd + (1 - gamma) * Pxx

    V_w = Pxx - alpha * Pdd
    P_ss = np.where(V_w > beta * Pdd, V_w, beta * Pdd)

    phase = np.exp(1j * np.angle(x_spectrum))
    s_complex = np.fft.ifft(np.sqrt(P_ss) * phase)
    
    return np.real(s_complex)[:frame_size], (noise_acc, Pdd, counter + 1)

def process_with_windows(signal, process_func, sr, 
                         window_ms=20, overlap_ratio=0.5, **kwargs):
    frame_size = int(sr * window_ms / 1000)
    hop_size = int(frame_size * (1 - overlap_ratio))
    
    window_func = np.hanning(frame_size)
    output = np.zeros_like(signal)
    weight_sum = np.zeros_like(signal) + 1e-10
    
    counter = 0
    nfft_init = 2 * (2 ** nextpow2(frame_size))
    noise_acc = np.zeros(nfft_init)
    Pdd = np.zeros(nfft_init)
    
    for start in range(0, len(signal) - frame_size + 1, hop_size):
        end = start + frame_size
        segment = signal[start:end] * window_func
        
        processed, (noise_acc, Pdd, counter) = process_func(
            segment, frame_size, counter, noise_acc, Pdd, **kwargs
        )
        
        output[start:end] += processed * window_func
        weight_sum[start:end] += window_func ** 2
    
    return output / weight_sum


def plot_spectrograms(clean_signal, noisy_signal, denoised_signal, sr, output_prefix):
    fs_spec = sr
    nperseg = min(1024, sr // 20)
    noverlap = nperseg // 2
    
    f1, t1, Sxx_clean = spectrogram(clean_signal, fs=fs_spec, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='spectrum')
    f2, t2, Sxx_noisy = spectrogram(noisy_signal, fs=fs_spec, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='spectrum')
    f3, t3, Sxx_denoised = spectrogram(denoised_signal, fs=fs_spec, nperseg=nperseg, noverlap=noverlap, window='hann', scaling='spectrum')
    
    Sxx_clean_db = 10 * np.log10(Sxx_clean + 1e-12)
    Sxx_noisy_db = 10 * np.log10(Sxx_noisy + 1e-12)
    Sxx_denoised_db = 10 * np.log10(Sxx_denoised + 1e-12)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=100)
    
    im1 = axes[0].pcolormesh(t1, f1, Sxx_clean_db, shading='gouraud', cmap='viridis')
    axes[0].set_ylabel('Frequency [Hz]')
    axes[0].set_title('Clean Signal')
    axes[0].set_ylim(0, sr // 2)
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')
    
    im2 = axes[1].pcolormesh(t2, f2, Sxx_noisy_db, shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency [Hz]')
    axes[1].set_title('Noisy Signal')
    axes[1].set_ylim(0, sr // 2)
    plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')
    
    im3 = axes[2].pcolormesh(t3, f3, Sxx_denoised_db, shading='gouraud', cmap='viridis')
    axes[2].set_ylabel('Frequency [Hz]')
    axes[2].set_xlabel('Time [sec]')
    axes[2].set_title('Denoised Signal')
    axes[2].set_ylim(0, sr // 2)
    plt.colorbar(im3, ax=axes[2], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_prefix + '_spectrograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig_diff, ax_diff = plt.subplots(1, 1, figsize=(10, 4), dpi=100)
    Sxx_diff = Sxx_denoised_db - Sxx_noisy_db
    im_diff = ax_diff.pcolormesh(t3, f3, Sxx_diff, shading='gouraud', cmap='RdBu_r', vmin=-20, vmax=20)
    ax_diff.set_ylabel('Frequency [Hz]')
    ax_diff.set_xlabel('Time [sec]')
    ax_diff.set_title('Spectral Difference (Denoised - Noisy) [dB]')
    ax_diff.set_ylim(0, sr // 2)
    plt.colorbar(im_diff, ax=ax_diff, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_prefix + '_difference.png', dpi=150, bbox_inches='tight')
    plt.close()

def test_denoising_pipeline(input_clean_path, output_denoised_path, 
                            noise_std=0.05, seed=42,
                            window_ms=20, overlap_ratio=0.5,
                            N_tres=1.1, alpha=5.1, beta=0.002, gamma=0.9,
                            plot_specs=True):
    sr, data = wav.read(input_clean_path)
    
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        clean_signal = data.astype(np.float64) / max_val
    else:
        clean_signal = data.astype(np.float64)
        max_val = 1.0
    
    noisy_signal = add_gaussian_noise_simple(clean_signal, noise_std=noise_std, seed=seed)
    
    denoised_signal = process_with_windows(
        noisy_signal, processfunc, sr,
        window_ms=window_ms, overlap_ratio=overlap_ratio,
        N_tres=N_tres, alpha=alpha, beta=beta, gamma=gamma
    )
    
    denoised_clipped = np.clip(denoised_signal, -1.0, 1.0)
    if np.issubdtype(data.dtype, np.integer):
        save_data = (denoised_clipped * max_val).astype(data.dtype)
    else:
        save_data = denoised_clipped.astype(data.dtype)
    
    wav.write(output_denoised_path, sr, save_data)
    
    if plot_specs:
        import os
        base_name = os.path.splitext(output_denoised_path)[0]
        plot_spectrograms(clean_signal, noisy_signal, denoised_clipped, sr, base_name)

if __name__ == "__main__":
    INPUT_CLEAN = r"/media/files/BSUIR/Labs/DAP/Lab4/05.wav"
    OUTPUT_DENOISED = r"/media/files/BSUIR/Labs/DAP/Lab4/05_denoised.wav"
    
    test_denoising_pipeline(
        INPUT_CLEAN, OUTPUT_DENOISED,
        noise_std=0.05, seed=42,
        window_ms=20, overlap_ratio=0.5,
        N_tres=1.1, alpha=5.1, beta=0.002, gamma=0.9,
        plot_specs=True
    )