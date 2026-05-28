from typing import List
from pathlib import Path

import random
import numpy as np
import librosa as lr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import WaveformResNet
from losses import FocalLoss
from temperature import learn_temperature

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

SR = 8000
n_fft = 512
n_mels = 128

# Define labels
labelpos = 1.0
labelneg = 0.0

def load_siren_dataset(
    balanced=False, 
    target_ratio=1.0,
    anchor="siren"
) -> List:
    '''
    Docstring for load_siren_dataset
    
    :param balanced: Description
    :param target_ratio: Ratio of siren to nonsiren samples after balancing
    :param anchor: "siren" to keep all sirens, "nonsiren" to keep all nonsirens

    :return: List of (file_path, label) tuples
    '''
    sirennet_dir = Path(__file__).parent.parent.parent / "dataset/sirennet/"
    idmt_dir     = Path(__file__).parent.parent.parent / "dataset/idmt_traffic/"

    # Get datafiles
    sirennet_all = list(sirennet_dir.glob("*/*.wav"))
    sirennet_nonsiren = list(sirennet_dir.glob("traffic/*.wav"))
    idmt_nonsiren = list(idmt_dir.glob("*.wav"))

    # Separate datafiles
    sirennet_siren = [f for f in sirennet_all if f not in sirennet_nonsiren]
    nonsiren_files = sirennet_nonsiren + idmt_nonsiren

    dataset_entries = (
        [(f, labelpos) for f in sirennet_siren] +
        [(f, labelneg) for f in nonsiren_files]
    )

    if not balanced:
        return dataset_entries

    # count current ratio
    n_siren = len(sirennet_siren)
    n_non   = len(nonsiren_files)

    if anchor == "siren":
        # keep all sirens, adjust nonsirens
        desired_siren = n_siren
        desired_non   = int(desired_siren / target_ratio) if target_ratio > 0 else n_non

        if desired_non <= n_non:
            # undersample nonsiren
            nons = random.sample(nonsiren_files, desired_non)
        else:
            # oversample nonsiren
            extra = random.choices(nonsiren_files, k=desired_non - n_non)
            nons = nonsiren_files + extra

        return (
            [(f, labelpos) for f in sirennet_siren] +
            [(f, labelneg) for f in nons]
        )

    elif anchor == "nonsiren":
        # keep all nonsirens, adjust sirens
        desired_non   = n_non
        desired_siren = int(target_ratio * desired_non)

        if desired_siren <= n_siren:
            # undersample sirens
            srs = random.sample(sirennet_siren, desired_siren)
        else:
            # oversample sirens
            extra = random.choices(sirennet_siren, k=desired_siren - n_siren)
            srs = sirennet_siren + extra

        return (
            [(f, labelpos) for f in srs] +
            [(f, labelneg) for f in nonsiren_files]
        )

    else:
        raise ValueError("anchor must be 'siren' or 'nonsiren'")

def augment_waveform(y, p, sr=SR):
    # Add random noise
    if random.random() < p[0]:
        noise_amp = np.random.uniform(0.001, 0.01)
        y = y + noise_amp * np.random.normal(size=y.shape)

    # Time stretching
    if random.random() < p[1]:
        speed_factor = random.uniform(0.98, 1.02)
        y = lr.effects.time_stretch(y, rate=speed_factor)

    # Pitch Shift
    if random.random() < p[2]:
        pitch = np.random.uniform(-0.2, 0.2)
        y = lr.effects.pitch_shift(y, sr=sr, n_steps=pitch)

    # Loudness
    db_change = np.random.randint(-50, 0)
    factor = 10 ** (db_change / 20)
    y = y * factor

    return y

# Generate synthetic waveform of random noise
def generate_noisy_waveform(SR=8000, duration=4):
    noise_amp = np.random.uniform(0.01, 0.05)
    return noise_amp * np.random.randn(int(SR * duration))

# Conversion to mel-spectrogram
def to_spectrogram(
    input_signal: np.ndarray,
    sr=8000, 
    n_fft=512, 
    hop_length=128
) -> np.ndarray:
    S = lr.stft(input_signal, n_fft=n_fft, hop_length=hop_length, center=False)
    mel_basis = lr.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S_mel = np.dot(mel_basis, np.abs(S))
    S_db = lr.amplitude_to_db(S_mel, ref=np.max)
    return S_db # [n_mels, T]

# Pad the waveform
def pad_waveform(
    y_list: List[np.ndarray], 
    target_length: int = None
) -> List[np.ndarray]:
    """
    Pad a list of waveforms to the same length.
    
    Args:
        y_list: list of 1D numpy arrays (waveforms)
        target_length: int, length to pad to. If None, uses the max length in the list.
    
    Returns:
        padded_list: list of 1D numpy arrays, all with length == target_length
    """
    if target_length is None:
        target_length = max(y.shape[0] for y in y_list)

    padded_list = []
    for y in y_list:
        if len(y) < target_length:
            pad_width = target_length - len(y)
            y_padded = np.pad(y, (0, pad_width), mode='constant')
        else:
            y_padded = y[:target_length]  # truncate if too long
        padded_list.append(y_padded)

    return padded_list

# Pad the spectrogram
def pad_spectrograms(S_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Pads a list of spectrograms to the same number of frames.
    Works for a single batch; does not require knowing the max length in the dataset.
    """
    # Find max mel bins and max frames
    max_mels = max(s.shape[0] for s in S_list)
    max_frames = max(s.shape[1] for s in S_list)

    padded_S = []
    for S in S_list:

        # Pad mel bins (frequency axis)
        if S.shape[0] < max_mels:
            pad_mel = np.zeros((max_mels - S.shape[0], S.shape[1]))
            S = np.concatenate([S, pad_mel], axis=0)

        # Pad frames (time axis)
        if S.shape[1] < max_frames:
            pad_frame = np.zeros((S.shape[0], max_frames - S.shape[1]))
            S = np.concatenate([S, pad_frame], axis=1)

        # print(f"Padded spectrogram shape: {S.shape}")
        padded_S.append(S)

    return padded_S


class SirenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: List,
        labels: List,
        sampling_rate: int,
        augmentation_probs_siren: List[float],
        augmentation_probs_nonsiren: List[float],
        num_augmentations_siren: int,
        num_augmentations_nonsiren: int,
        add_original: bool = True,
        num_noisy_samples: int = 0,
        n_fft: int = 512
    ):
        # Siren: 1, Nonsiren: 0
        self.samples = samples
        self.labels = labels
        self.sr = sampling_rate
        self.aug_probs_siren = augmentation_probs_siren
        self.aug_probs_nonsiren = augmentation_probs_nonsiren
        self.n_augs_siren = num_augmentations_siren
        self.n_augs_nonsiren = num_augmentations_nonsiren
        self.num_noisy_samples = num_noisy_samples
        self.add_original = add_original
        self.n_fft = n_fft

        # Add number of original and augmentations to index
        self.index = []
        for sample, label in zip(samples, labels):
            if self.add_original:
                self.index.append((sample, label))
            n_aug = self.n_augs_siren if label == labelpos else self.n_augs_nonsiren
            for _ in range(n_aug):
                self.index.append((sample, label, True))

        # Add number of noisy waveforms query to index
        for _ in range(self.num_noisy_samples):
            self.index.append(("synthetic_noise", labelneg))
        
        # Number of positive / negative samples
        self.n_pos = sum(1 for p in self.labels if p == labelpos)
        self.n_neg = sum(1 for p in self.labels if p == labelneg)
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path, label, *aug_flag = self.index[idx]

        waveforms = []

        # Generate synthetic noisy waveform
        if path == "synthetic_noise":
            waveforms.append(generate_noisy_waveform())
            return waveforms, label

        try:
            waveform, _ = lr.load(path, sr=self.sr)
        except Exception as e:
            # Not bad if done minimally. Introduces duplicates to training.
            new_idx = random.randint(0, len(self.index) - 1)
            return self.__getitem__(new_idx)

        # Add original
        if self.add_original:
            waveforms.append(waveform)

        # Add augmented
        if label == 1:
            for _ in range(self.n_augs_siren):
                waveforms.append(
                    augment_waveform(waveform, self.aug_probs_siren, self.sr)
                )
        else:
            for _ in range(self.n_augs_nonsiren):
                waveforms.append(
                    augment_waveform(waveform, self.aug_probs_nonsiren, self.sr)
                )

        return waveforms, label

def siren_collate_fn(batch, sr=SR, n_fft=n_fft):
    # Uses global variables SR and n_fft
    waveforms = []
    labels = []

    for specs, label in batch:
        waveforms.extend(specs)
        labels.extend([label] * len(specs))

    # Pad waveforms
    target_length = max(len(y) for y in waveforms)
    padded_waveforms = pad_waveform(waveforms, target_length)

    # Convert to mel-spectrograms
    specs = [to_spectrogram(x, sr=sr, n_fft=n_fft) for x in padded_waveforms]

    # Convert to tensors
    X = torch.stack([torch.tensor(s, dtype=torch.float32) for s in specs])
    Y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    return X, Y

# Initialize loss functions
def bce_logitloss(preds, targets):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(preds, targets)

def focal_loss(preds, targets, *args):
    criterion = FocalLoss(alpha=0.50, gamma=2.0)
    return criterion(preds, targets)

# -----------------------------------------------------------------
#                          Training
# -----------------------------------------------------------------
# Pipeline: Load data, Augment data, Convert data, Pad data, Set dataloader, ...

# Load dataset entries
dataset_entries = load_siren_dataset(balanced=True, target_ratio=0.17, anchor="siren")

# Print class distribution
unique, counts = torch.unique(torch.tensor([label for _, label in dataset_entries]), return_counts=True)
print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

# Initialize dataloaders
train_entries, val_entries = train_test_split(
    dataset_entries,
    test_size=0.2,
    stratify=[label for _, label in dataset_entries],
    random_state=seed
)
X_train, Y_train = zip(*train_entries)
X_val, Y_val = zip(*val_entries)

train_dataset = SirenDataset(
    samples=X_train,
    labels=Y_train,
    sampling_rate=SR,
    augmentation_probs_siren=[1.0, 1.0, 1.0],
    augmentation_probs_nonsiren=[1.0, 1.0, 1.0],
    num_augmentations_siren=5,
    num_augmentations_nonsiren=0,
    num_noisy_samples=250,
    add_original=True,
    n_fft=n_fft
)
val_dataset = SirenDataset(
    samples=X_val,
    labels=Y_val,
    sampling_rate=SR,
    augmentation_probs_siren=[1.0, 1.0, 1.0],
    augmentation_probs_nonsiren=[1.0, 1.0, 1.0],
    num_augmentations_siren=1,
    num_augmentations_nonsiren=0,
    num_noisy_samples=250,
    add_original=True,
    n_fft=n_fft
)

batch_size = 64
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    collate_fn=siren_collate_fn
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    collate_fn=siren_collate_fn
)

# Print class distribution post-augmentation
total_pos = train_dataset.n_pos * (train_dataset.add_original + train_dataset.n_augs_siren)
total_neg = train_dataset.n_neg * (train_dataset.add_original + train_dataset.n_augs_nonsiren) + train_dataset.num_noisy_samples
print(f"Training (total including augmentations): Sirens {total_pos}, Nonsirens {total_neg}")
total_pos_ = val_dataset.n_pos * (val_dataset.add_original + val_dataset.n_augs_siren)
total_neg_ = val_dataset.n_neg * (val_dataset.add_original + val_dataset.n_augs_nonsiren) + val_dataset.num_noisy_samples
print(f"Validation (total including augmentations): Sirens {total_pos_}, Nonsirens {total_neg_}")


# Model initialization
n_epochs = 30
n_mels, n_bins = next(iter(train_loader))[0].shape[1:3]
print(f"n_bins for this training session: {n_bins}")
# n_bins = 247
model = WaveformResNet(
    bin_shape=n_bins, 
    mel_shape=n_mels,
    hidden_channels=48,
    n_resBlocks=3,
    gru_hidden=48,
    gru_layers=3, 
    attn_hidden=32,
    dropout=0.4
)
# model.load_state_dict(torch.load("waveform_model.pth", weights_only=True))

# Initialize optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-3,
    weight_decay=1e-1,
    betas=(0.80, 0.99)
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
    eta_min=3e-6
)

# Total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)

# Training
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    logits = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = bce_logitloss(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
        logits.append(out.detach().view(-1))

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            loss = bce_logitloss(out, yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(val_loader.dataset)

    cosine_scheduler.step()
    print(f"\033[33mEpoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\033[0m")
    
    all_logits = torch.cat(logits)
    print(f"Min/Max/Mean logits: {all_logits.min().item():.3f}, {all_logits.max().item():.3f}, {all_logits.mean().item():.3f}")

torch.save(model.state_dict(), "waveform_model.pth")

# -----------------------------------------------------------------
#                          Evaluation
# -----------------------------------------------------------------

# Find temperature for scaling logits
T = learn_temperature(model, val_loader)
print("Learned Temperature:", T)

# Evaluation
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

probs_list = []
y_list = []

logits_min = 1e9; 
logits_max = -1e9; 
logits_sum = 0.0; 
logits_sq = 0.0; 
n_tot = 0

model.eval()
with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb)
        probs = torch.sigmoid(logits / T)

        # Accumulate predictions
        probs_list.append(probs.cpu().numpy())
        y_list.append(yb.view(-1).cpu().numpy())

        lmin = logits.min().item()
        lmax = logits.max().item()
        logits_min = min(logits_min, lmin)
        logits_max = max(logits_max, lmax)
        logits_sum += logits.sum().item()
        logits_sq += (logits**2).sum().item()
        n_tot += logits.numel()

probs = np.concatenate(probs_list)
y_true = np.concatenate(y_list).astype(int)

# Find sufficient threshold other than 0.0 for evaluation
sorted_probs = np.sort(probs)
thresholds = np.unique(sorted_probs)
best_thr, best_f1 = 0.0, -1.0

for t in thresholds:
    preds_t = (probs > t).astype(int)
    f1_t = f1_score(y_true, preds_t, zero_division=0)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thr = t
preds = (probs > best_thr).astype(int)
print(f"Optimal F1-Score: {best_f1:.3f} found at Threshold: {best_thr:.4f}")

acc = accuracy_score(y_true, preds)
prec = precision_score(y_true, preds, zero_division=0)
rec = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)
mcc = matthews_corrcoef(y_true, preds)
cm = confusion_matrix(y_true, preds)
auc = roc_auc_score(y_true, probs)
frac_pos = float(preds.mean())

logits_mean = logits_sum / n_tot
logits_std = (logits_sq / n_tot - logits_mean**2)**0.5

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"MCC:       {mcc:.3f}")
print(f"CM:        {cm[0].tolist()}, {cm[1].tolist()}")
print(f"AUC:       {auc:.3f}")
print(f"Threshold: {best_thr:.2f}")
print(f"frac_pos:  {frac_pos:.2f}")
print(f"Logits_mean: {logits_mean:.3f}")
print(f"Logits_std:  {logits_std:.3f}")