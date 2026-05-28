from pathlib import Path
import librosa as sound
import numpy as np

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix
)

seed = 142
np.random.seed(seed)

SR = 8000
n_fft = 512
n_mels = 128

def normalize(y):
    return y / np.max(np.abs(y) + 1e-6)

def to_spec(y, sr=8000, n_fft=512, hop_length=128) -> np.ndarray:
    S = sound.stft(y, n_fft=n_fft, hop_length=hop_length, center=False)
    mel_basis = sound.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S_mel = np.dot(mel_basis, np.abs(S))
    S_db = sound.amplitude_to_db(S_mel, ref=np.max)
    return S_db # [n_mels, T]

def pad_spectrograms(S_list: list):
    # Find the longest time dimension
    max_frames = max(S.shape[1] for S in S_list)
    
    padded = []
    for S in S_list:
        if S.shape[1] >= max_frames:
            S_p = S[:, :max_frames]
        else:
            pad = np.zeros((S.shape[0], max_frames - S.shape[1]))
            S_p = np.concatenate([S, pad], axis=1)
        padded.append(torch.tensor(S_p, dtype=torch.float32))
    
    # Stack into a single tensor: [num_samples, freq, time]
    return torch.stack(padded)

def get_dataset(samples: int=50):
    def generate_noisy():
        duration = 4
        return np.random.randint(-5, 5) * np.random.randn(int(SR * duration))
    waveforms = [generate_noisy() for _ in range(samples)]
    # sf.write("noisywaveform.wav", waveforms[0], SR)
    return waveforms

# ------------------------------------------ #

nonsiren_dataset = list((Path(__file__).parent.parent.parent / "dataset/ESC-50/").glob("CarHorn/*.ogg"))
siren_dataset = list((Path(__file__).parent.parent.parent / "dataset/ESC-50/").glob("Siren/*.ogg"))

siren_waveforms = [sound.load(siren, sr=SR)[0] for siren in siren_dataset]
# siren_waveforms = [normalize(y) for y in siren_waveforms]
siren_waveforms = [to_spec(x, n_fft) for x in siren_waveforms]

nonsiren_waveforms = [sound.load(nonsiren, sr=SR)[0] for nonsiren in nonsiren_dataset]
nonsiren_waveforms += get_dataset() # Add synthetic dataset
# nonsiren_waveforms = [normalize(y) for y in nonsiren_waveforms]
nonsiren_waveforms = [to_spec(x, n_fft) for x in nonsiren_waveforms]

waveforms = siren_waveforms + nonsiren_waveforms

X_tensors = pad_spectrograms(waveforms)
Y_tensors = torch.tensor([1] * len(siren_waveforms) + [0] * len(nonsiren_waveforms), dtype=torch.float32)  # 1 = siren, 0 = nonsiren
dataset = TensorDataset(X_tensors, Y_tensors.unsqueeze(1))

# Load dataloader
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load model
from model import WaveformResNet
n_bins = 247
model = WaveformResNet(
    bin_shape=n_bins, 
    mel_shape=n_mels,
    hidden_channels=48,
    n_resBlocks=3,
    gru_hidden=48,
    gru_layers=3, 
    attn_hidden=32,
    dropout=0.2,
    dropout_head=0.3
)
model.load_state_dict(torch.load("waveform_model.pth", weights_only=True))
model.eval()

y_list = []
probs_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        preds = torch.sigmoid(outputs)

        # Accumulate predictions
        probs_list.append(preds.cpu().numpy())
        y_list.append(y_batch.view(-1).cpu().numpy())

probs = np.concatenate(probs_list)
y_true = np.concatenate(y_list).astype(int)

# Find sufficient threshold other than 0.0 for evaluation
thresholds = np.linspace(0.0, 1.0, 20)
best_thr, best_f1 = 0.5, -1.0
for t in thresholds:
    preds_t = (probs > t).astype(int)
    f1_t = f1_score(y_true, preds_t, zero_division=0)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thr = t
y_pred = (probs > best_thr).astype(int)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=[0,1])

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"MCC:       {mcc:.4f}")
print(f"CM:        {cm[0].tolist()}, {cm[1].tolist()}") # [TN, FP], [FN, TP]

# How confident is the model with its predictions?
pred_pos_mask = (y_pred == 1)
pred_pos_probs = probs[pred_pos_mask]
print(f"Positive prediction prob mean: {pred_pos_probs.mean()}")
print(f"Positive prediction prob var:  {pred_pos_probs.std()}")
print()

# Is the model just confident and lucky?
prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability diagram')
plt.legend()
plt.grid(True)
plt.show()
