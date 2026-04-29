import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.fftpack as fft
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from pathlib import Path

src_dir = Path(__file__).parent
a_dir = src_dir / "Dataset/class_A"
b_dir = src_dir / "Dataset/class_B"

SR = 16000
DURATION = 2
TARGET_LEN = DURATION * SR
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 15

font_dir = ['/usr/share/fonts/TTF/']
for font_file in fm.findSystemFonts(fontpaths=font_dir):
    fm.fontManager.addfont(font_file)

font_label = {'fontname': 'Times New Roman', 'size': 12}
font_title = {'fontname': 'Times New Roman', 'size': 14, 'weight': 'bold'}

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
        class_files = [f for f in class_dir.glob('*.wav')][:10]
        for path in class_files:
            audio, _ = librosa.load(path, sr=SR, mono=True)
            features = extract_features(audio, use_custom=use_custom, add_std=add_std)
            X.append(features)
            y.append(class_idx)
            files.append(path.name)
    return np.array(X), np.array(y), files

X, y, file_names = load_dataset(use_custom=False, add_std=False)

X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, file_names, test_size=4, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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
plt.title('Confusion Matrix', **font_title)
plt.xlabel('Predicted', **font_label)
plt.ylabel('True', **font_label)
plt.show()

X_30, y_30, _ = load_dataset(use_custom=False, add_std=True)
X_tr_30, X_te_30, y_tr_30, y_te_30 = train_test_split(
    X_30, y_30, test_size=4, stratify=y_30, random_state=42)

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
plt.plot(range(1, 9), accuracies, marker='o', linestyle='-', color='steelblue')
plt.xlabel('k', **font_label)
plt.ylabel('Accuracy', **font_label)
plt.title('Accuracy to k', **font_title)
plt.xticks(range(1, 9))
plt.grid(True, alpha=0.3)
plt.show()

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3, metric='euclidean'))
])

loo = LeaveOneOut()
scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')
print(f"LOO Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")