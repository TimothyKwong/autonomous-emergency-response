# Autonomous Audio-Photonic Emergency Vehicle Response

## Project Overview

### Description
This course project develops an autonomous system for emergency vehicle response. The system processes acoustic and photonic data to detect emergency vehicles, estimate their motion, and determine the real-time ego-response (stop or go).

### Objectives
- Data Acquisition: Capture ambient audio using on-board microphone and strobing lights using the ESP32 photoresistor.
- Signal Processing: Transform raw audio waveforms from time-domain to time-frequency representations using with Short-Time Fourier Transform (STFT).
- Siren Classification: Ingest time-frequency representations for binary classification on emergency vehicle sirens (ambulances, fire trucks, police cars) versus non-emergency acoustic signals.
- Strobing Light Classification: Model and recognize the simple, fixed flicker frequency.
- Multimodal Data Fusion: Integrate siren and strobing light classifications to verify the presence of an emergency vehicle.
- Motion State Estimation: Tracked changes in the slope of the dominant audio frequency over time to estimate an approaching, centered, or passing state.
- Autonomous Decision Making: Apply simple decision tree using classifications and estimated motion states to determine "Stop" or "Go" action.

### System Inputs & Sensor Integration
The system acquires environmental data using two main hardware sensors:
| Sensor Type | Input | Purpose |
| :--- | :--- | :--- |
| **Microphone** | Acoustics | Captures raw audio frequencies from sirens |
| **Photoresistor** | Strobing Lights | Detects the distinct flickering frequencies of emergency lights |

## Directory Structure

```
├── checkpoints/
├── data/
├── dataset/
├── figures/
├── src/
│   ├── nn/                       # Neural network modules
│   │   ├── model.py              # WaveformResNet architecture
│   │   ├── train.py              # Training and evaluation pipeline
│   │   ├── losses.py             # Custom loss functions
│   │   ├── evaluation.py         # Evaluation metrics
│   │   ├── temperature.py        # Temperature scaling for calibration
│   │   └── conversion.py         # Model conversion utilities
│   ├── mic.py                    # Microphone interface for audio acquisition
│   ├── sensor.py                 # Sensor interface and threading
│   └── logger.py                 # Logging utilities
├── .gitignore
├── .gitkeep
├── main.ino
├── README.md
└── recog.py                      # Main recognition pipeline
```

## Requirements

### Hardware Requirements
- **ESP32 Microcontroller**: This code requires an ESP32 board or similar for real-time photoresistor sensor data acquisition and data transmission via USB.

### Software Requirements
- Python 3.9+
- PyTorch
- NumPy
- Librosa
- SciPy
- Matplotlib
- PySerial
- sounddevice
- soundfile
- scikit-learn

## Installation

```bash
# Install dependencies
pip install torch numpy scipy scikit-learn librosa sounddevice soundfile pyserial matplotlib
```

## Usage

### Training the model

Run the training pipeline for the neural network using:

```bash
python -m src.nn.main
```

### Main execution

Run the following script to execute the complete pipeline from raw sensor data acquisition to control decision:

```bash
python recog.py
```

## Technical Details

### Datasets
Source repositories for training and evaluation data:
- **sireNNet:** Audio dataset for siren detection [[Mendeley Data Link](https://data.mendeley.com/datasets/j4ydzzv4kb/1)]
- **IDMT_Traffic:** Traffic noise dataset for acoustic scene analysis [[Zenodo Link](https://zenodo.org/records/7551553)]
- **ESC-50:** Environmental Sound Classification repository, filtering for categories like *Siren*, *Car Horn*, and *Car Engine* [[GitHub Link](https://github.com/karolpiczak/ESC-50)]

### Neural Network Architecture
- Illustration of architecture can be found in [figures/ppt_figures.png](figures/ppt_figures.png)

### Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Error Analysis:** Confusion Matrix, Calibration Curve

## Limitations

- Significant performance degradation in model accuracy during real-time deployment, likely caused by distribution shift and weak generalization.
- Simplified detection by assuming emergency vehicles use a simple, fixed flicker frequency and white light.