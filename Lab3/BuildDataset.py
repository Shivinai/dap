import os
import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000  
TARGET_DURATION = 2 
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION  

def BuildDataset(input_dir, output_dir):

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")
            
        try:
            y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

            if len(y) > TARGET_SAMPLES:
                y = y[:TARGET_SAMPLES]
            else:
                y = np.pad(y, (0, TARGET_SAMPLES - len(y)))

            sf.write(output_path, y, TARGET_SR, subtype='PCM_16')
            print(f"Processed: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

BuildDataset("Lab3/Data/birb", "Lab3/Dataset/class_A")

BuildDataset("Lab3/Data/wind", "Lab3/Dataset/class_B")

print("\nDone")