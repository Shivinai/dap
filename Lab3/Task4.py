import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from pathlib import Path

src_dir = Path(__file__).parent
a_dir = os.path.join(src_dir, "Dataset/class_A")
b_dir = os.path.join(src_dir, "Dataset/class_B")

SR = 16000
DURATION = 2
TARGET_LEN = DURATION * SR
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 15

def load_and_pad(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    return y

def custom_mfcc(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC):
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2

    S = librosa.feature.melspectrogram(S=D, sr=sr, n_fft=n_fft, hop_length=hop_length)

    S_dB = librosa.power_to_db(S, ref=np.max)

    mfccs = fft.dct(S_dB, type=2, axis=0, norm='ortho')[:n_mfcc]
    return mfccs

def extract_features(y, use_custom=False, add_std=False):
    if use_custom:
        mfccs = custom_mfcc(y)
    else:
        mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    vec_mean = np.mean(mfccs, axis=1)
    
    if add_std:
        vec_std = np.std(mfccs, axis=1)
        return np.concatenate([vec_mean, vec_std])
    return vec_mean

def load_dataset(use_custom=False, add_std=False):
    X, y, files = [], [], []

    for class_idx, class_dir in enumerate([a_dir, b_dir]):
        class_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        for fname in class_files[:10]:
            path = os.path.join(class_dir, fname)
            audio = load_and_pad(path)
            features = extract_features(audio, use_custom=use_custom, add_std=add_std)
            
            X.append(features)
            y.append(class_idx)
            files.append(fname)
            
    return np.array(X), np.array(y), files

X, y, file_names = load_dataset(use_custom=False, add_std=False)

if len(X) > 0:
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(X, y, file_names, test_size=4, stratify=y, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    y_proba = knn.predict_proba(X_test_scaled)
    
    classes_names = ['Class A', 'Class B']
    for fname, pred, prob in zip(files_test, y_pred, y_proba):
        print(f"Filename: {fname} | Predicted: {classes_names[pred]} | Probability: [A: {prob[0]:.3f}, B: {prob[1]:.3f}]")
        
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2%}")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    X_30, y_30, _ = load_dataset(use_custom=False, add_std=True)
    X_tr_30, X_te_30, y_tr_30, y_te_30 = train_test_split(X_30, y_30, test_size=4, stratify=y_30, random_state=42)
    
    scaler_30 = StandardScaler()
    X_tr_scaled_30 = scaler_30.fit_transform(X_tr_30)
    X_te_scaled_30 = scaler_30.transform(X_te_30)
    
    knn_30 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn_30.fit(X_tr_scaled_30, y_tr_30)
    acc_30 = accuracy_score(y_te_30, knn_30.predict(X_te_scaled_30))
    print(f"Accuracy on 30 factors: {acc_30:.2%}")

    accuracies = []
    for k in range(1, 9):
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        model.fit(X_train_scaled, y_train)
        accuracies.append(accuracy_score(y_test, model.predict(X_test_scaled)))
        
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 9), accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 9))
    plt.grid(True)
    plt.show()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=3, metric='euclidean'))
    ])
    
    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')
    print(f"LOO Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")