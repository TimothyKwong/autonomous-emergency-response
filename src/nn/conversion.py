# Found at github/esp-dl > examples > tutorial > how_to_quantize_model > quantize_sin_model

# python3 src/main.py
# python3 src/evaluation.py
# python3 src/conversion.py
# xxd -i waveform_model.espdl > waveform_model.h

import torch
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_torch, load_graph
from esp_ppq.executor.torch import TorchExecutor
from model import WaveformResNet

SAMPLE_RATE = 8000
DEVICE = "cpu"
BATCH_SIZE = 1
CALIB_STEPS = 48
ESPDL_MODEL_PATH = "waveform_model.espdl"
TARGET = "esp32s3"
NUM_OF_BITS = 8
INPUT_SHAPE = [1, 1, SAMPLE_RATE]

train_dataset = torch.load("X_train.pt", weights_only=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model_fp32 = WaveformResNet(hidden_channels=18, seq_len=SAMPLE_RATE, n_resBlocks=3)
model_fp32.load_state_dict(torch.load("waveform_model.pth", weights_only=True))
model_fp32.eval()

def collate_fn(batch):
    # torch.Size([16, 1, 8000])
    return batch

quant_ppq_graph = espdl_quantize_torch(
    model=model_fp32,
    espdl_export_file=ESPDL_MODEL_PATH,
    calib_dataloader=train_loader,
    calib_steps=CALIB_STEPS,
    input_shape=INPUT_SHAPE,
    target=TARGET,
    num_of_bits=NUM_OF_BITS,
    device=DEVICE,
    error_report=True,
    skip_export=False,
    export_test_values=True,
    verbose=1,
)

# ----------------------------------
# --- Validation of Quantization ---
# ----------------------------------
executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)

y_train = torch.load("y_train.pt", weights_only=True)
dataset = TensorDataset(train_dataset, y_train)
train_loader_ = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

criterion = torch.nn.BCEWithLogitsLoss()
loss = 0.0

with torch.no_grad():
    for batch_x, batch_y in train_loader_:
        y_pred = executor(batch_x)
        loss += criterion(y_pred[0], batch_y)

loss /= len(train_loader_)
print(f"Quantized model loss: {loss.item():.5f}")