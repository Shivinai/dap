import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def RMS_level(alpha, beta, x_sample, p_sample):
    if (x_sample**2 >= p_sample):
        p = alpha * p_sample + (1 - alpha) * x_sample**2
    else:
        p = beta * p_sample + (1 - beta) * x_sample**2

    p = max(p, 1e-12)

    log_p = 10 * np.log10(p)
    return log_p


def interpolate_3points(point1, point2, point3, x):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    if x <= x2:
        if x2 - x1 != 0:
            t = (x - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
        else:
            y = y1
    else:
        if x3 - x2 != 0:
            t = (x - x2) / (x3 - x2)
            y = y2 + t * (y3 - y2)
        else:
            y = y2
    
    return y


alpha = 0.001511
beta = 0.0004535

input_path = r"Eric_Johnson_-_S.R.V._(SkySound.cc).wav"
output_path = r"output.wav"

sample_rate, audio_data = wavfile.read(input_path)

audio_data = audio_data.mean(axis=1)

original_dtype = audio_data.dtype

audio_data = audio_data.astype(np.float32)

if original_dtype == np.int16:
    audio_data = audio_data / 32768
elif original_dtype == np.int32:
    audio_data = audio_data / 2147483648

p_level = 10**(-5)

point1 = (-90, -90)
point2 = (-20, -45)
point3 = (0, -25)

for i in range(len(audio_data)):
    p_prev = p_level
    p_level = RMS_level(alpha, beta, audio_data[i], p_prev)

    new_level = interpolate_3points(point1, point2, point3, p_level)

    G = new_level - p_level
    g = 10**(G / 20)

    audio_data[i] = audio_data[i] * g


max_val = np.max(np.abs(audio_data))
audio_data = audio_data / max_val
audio_data_out = (audio_data * 32767).astype(np.int16)

input_db = np.linspace(-100, 0, 1000)
output_db = [interpolate_3points(point1, point2, point3, x) for x in input_db]

wavfile.write(output_path, sample_rate, audio_data_out)

plt.figure(figsize=(8, 8))

plt.plot(input_db, output_db, 'b-', linewidth=2, label='Передаточная характеристика')

plt.plot([-100, 0], [-100, 0], 'k--', alpha=0.5, label='Прямой сигнал (1:1)')

for p in [point1, point2, point3]:
    plt.plot(p[0], p[1], 'ro')
    plt.annotate(f'({p[0]}, {p[1]})', (p[0], p[1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Передаточная характеристика компрессора (Input vs Output)')
plt.xlabel('Входной уровень (dB)')
plt.ylabel('Выходной уровень (dB)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.xlim([-100, 5])
plt.ylim([-100, 5])
plt.gca().set_aspect('equal', adjustable='box') 

plt.show()

f1,t1,S1 = spectrogram(audio_data, sample_rate)
f2,t2,S2 = spectrogram(audio_data_out, sample_rate)

plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.pcolormesh(t1,f1,10*np.log10(S1 + 1e-10), shading='gouraud')
plt.title("Good source music spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.subplot(2,1,2)
plt.pcolormesh(t2,f2,10*np.log10(S2 + 1e-10), shading='gouraud')
plt.title("Processed signal spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()