from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Union, Callable, Any

from collections import deque
from enum import Enum

from pathlib import Path
import csv

import serial
import time
import numpy as np
import torch

import librosa as lr
from scipy.signal import iirfilter, sosfilt

import matplotlib.pyplot as plt

from src.mic import MicInterface
from src.sensor import SensorInterface, SensorThread
from src.nn.model import WaveformResNet
from src.logger import MsgLogger

# ------------------ Initializaiton ------------------ #

@dataclass
class EmergencyVehicleLightModel:
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        # "non_emergency": (0.0, 1.0),
        "emergency":       (1.0, 6.0),
    })

class Vehicle(Enum):
    NONEMERGENCY = 0
    EMERGENCY    = 1

class Decision(Enum):
    GO   = "Go"
    STOP = "Stop"

# ------------------ Outlier Detection ------------------ #
# Mahalanobis distance
# Z-Score
# K-Nearest Neighbour
# Local Outlier Factor
# Random Sample Consensus (RANSAC)

def detectOutliers(input: np.ndarray) -> np.ndarray:
    """
    Instead of detecting outliers, we detect non-outliers
    to reduce their impact on outliers 
    (similar to signal-to-noise ratio).

    Modified Z-Score
    
    input: ndarray shape (N_samples)

    :return: Description
    """
    threshold = 5.0
    median = np.median(input)
    mad = np.median(np.abs(input - median))
    outlier_scores = np.abs(input - median) / (mad if mad != 0 else 1.0)
    outliers = outlier_scores > threshold
    return outliers

# ------------------ Signal Processing ------------------ #
def normalizeSignal(input: np.ndarray) -> np.ndarray:
    return input - np.mean(input)

def filterBandpass(
    input, 
    low_cutoff=100, 
    high_cutoff=3000, # TODO: high cutoff cannot be >3000
    sr=8000, 
    order=10
) -> np.ndarray:
    """ 
    Band-pass filter (time-domain)
    """
    sos = iirfilter(
        N=order,
        Wn=[low_cutoff, high_cutoff],
        btype='band',
        ftype='butter',
        fs=sr,
        output='sos'
    )
    return sosfilt(sos, input)

# ------------------ FFT ------------------ #
def performFFT(input_signal: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude FFT of a 1D signal.

    Parameters
    ----------
    input_signal : np.ndarray
        Time-domain signal of shape [N].

    Returns
    -------
    np.ndarray
        Magnitude spectrum of length N
    """
    N = len(input_signal)
    spectrum = np.fft.rfft(input_signal)
    magnitude = np.abs(spectrum) / N
    if N % 2 == 0:
        magnitude[1:-1] *= 2
    else:
        magnitude[1:] *= 2
    return magnitude

def performSTFT(
    input_signal: np.ndarray, 
    sr=8000, 
    n_fft=512, 
    hop_length=128
) -> np.ndarray:
    """
    Compute the STFT (short-time Fourier transform).

    Parameters
    ----------
    input_signal : np.ndarray
        Time-domain signal.
    n_fft : int
        FFT size per frame.
    hop_length : int
        Step between frames.

    Returns
    -------
    np.ndarray
        Complex STFT matrix of shape [(n_fft//2 + 1), num_frames].
    """
    S = lr.stft(
        input_signal, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        center=False
    )
    mel_basis = lr.filters.mel(
        sr=sr, 
        n_fft=n_fft, 
        n_mels=n_mels
    )
    S_mel = np.dot(mel_basis, np.abs(S)) # Discard i*Im from Re + i*Im
    S_db = lr.amplitude_to_db(S_mel, ref=np.max) # ref=np.max destroys loudness?
    return S_db

# ------------------ Inference ------------------ #
# Given that sampling rate are not guaranteed to be
# consistent, operations, such as FFT and STFT, that
# rely on sampling rate should be use sparingly. FFT
# and STFT assume a sampling rate with even periods.
# However, the board requires code to guarantee a 
# sampling rate with even periods by ... Otherwise,
# the board "brute-forces" sampling. (Speculation)
def inferLight(
    input_data: np.ndarray,
    duration_ms: np.ndarray,
    profile: EmergencyVehicleLightModel,
    outliers: Union[List, np.ndarray]
) -> bool:
    """ 
    Utilize the frequency at which the light turns on and off 
    (flash frequency) to infer the emergency vehicles.
    """
    # Sampling rate for this buffer
    duration = np.sum(duration_ms) / 1000

    # Add window if input_data has t>1s of data, otherwise lag.

    # Edge detection
    diff = np.diff(input_data)
    threshold = 0.5
    events = diff > threshold
    num_changes = np.sum(events)
    freq = num_changes / duration

    # For diagnostic purposes
    outlier_ratio = (np.sum(outliers) / np.sum(events)) if np.sum(events) != 0 else 0.0

    # Compare to profile
    low_f, high_f = profile.ranges["emergency"]
    if low_f <= freq <= high_f:
        return True

    return False


@torch.no_grad
def inferSound(
    input: np.ndarray, 
    audio_model: WaveformResNet,
    temperature: float
) -> np.ndarray:
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    logits = audio_model(input)
    probs = torch.sigmoid(logits / temperature)
    probs = probs.squeeze()
    probs = np.stack([1 - probs, probs], axis=-1) # [neg class, pos class]
    logger.collect(f"Non-Siren prob: {probs[0]:.3f}, Siren prob: {probs[1]:.3f}")
    return probs


class PassageBayesFilter:
    ''' 
    Objective: Find change in frequency over time

    Estimate state: approaching -> centered -> receding
    
    Hidden states:
        0 = approaching
        1 = centered
        2 = passing

    Observation:
        slope of dominant frequency over time
    '''

    def __init__(self):
        self.hidden_states = {
            0: "approaching",
            1: "centered",
            2: "passing"
        }

        # Initialize belief state
        self.belief = np.array([
            1/3, # Approaching
            1/3, # Centered
            1/3, # Passing
        ])

        # Gaussian transition prob
        self.T = np.array([
            [0.90, 0.10, 0.00],   # approaching
            [0.05, 0.90, 0.05],   # centered
            [0.00, 0.10, 0.90],   # receding
        ])

        self.mu = np.array([150.0, 0.0, -150.0])
        self.sigma = 150.0

    def _dominant_freq_slope(
        self,
        signal: np.ndarray, 
        sr=8000, 
        n_fft=512, 
        hop_length=128
    ) -> float:
        # Compute STFT, [n_mels, T]. (STFT is extra if passing in STFT)
        Z = lr.stft(
            signal, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        mag = np.abs(Z)

        # Number of time frames is sufficient for inference
        num_frames = mag.shape[1]
        if num_frames < 5:
            return 0.0
        
        # Dominant frequency per frame
        dom_freqs = []
        for i in range(num_frames):
            col = mag[:, i]
            dom_bin = np.argmax(col)
            dom_freq = lr.fft_frequencies(sr=sr, n_fft=n_fft)[dom_bin]
            dom_freqs.append(dom_freq)
        dom_freqs = np.array(dom_freqs)

        # Time axis for each frame (in seconds)
        t = lr.frames_to_time(
            np.arange(num_frames), 
            sr=sr, 
            hop_length=hop_length, 
            n_fft=n_fft
        )

        # Fit slope: freq vs time
        slope = np.polyfit(t, dom_freqs, 1)[0]

        return slope
    
    def _likelihood(self, slope: float) -> np.ndarray:
        ''' Gaussian likelihood: P(z | state) '''
        norm = 1 / np.sqrt(2 * np.pi * self.sigma ** 2)
        probs = norm * np.exp(-0.5 * ((slope - self.mu) / self.sigma) ** 2)
        return probs
    
    def _update(self, slope: np.ndarray):
        # Bayes Filter

        # transition * belief = p(x_k | x_k−1, v_k) * p(x_k-1 | . )
        prior_belief = self.T.T @ self.belief

        # observation = p(measurement | state)
        likelihood = self._likelihood(slope)

        # ...
        posterior_belief = likelihood * prior_belief
        Z = 1.0 / np.sum(posterior_belief)

        # Update parameters
        self.belief = Z * posterior_belief
    
    def _most_likely_state(self) -> int:
        return int(np.argmax(self.belief))

    def hasPassed(
        self, 
        signal: np.ndarray, 
        sr=8000, 
        n_fft=512, 
        hop_length=128
    ) -> bool:
        slope = self._dominant_freq_slope(signal, sr, n_fft, hop_length)
        
        self._update(slope)
        
        state_idx = self._most_likely_state()

        logger.collect(f"Belief state: {self.hidden_states[state_idx]}.")

        if self.hidden_states[state_idx] == "passing":
            return True
        else:
            return False

# ------------------ Information fusion ------------------ #
def booleanToProbs(b: bool) -> np.ndarray:
    if b:
        return np.array([0.0, 1.0], dtype=float)
    else:
        return np.array([1.0, 0.0], dtype=float)
    
class InformationFusion:
    def __init__(self, threshold=0.7):
        """
        p_stay: probability the emergency state persists from one step to next
        threshold: posterior threshold for declaring 'emergency'
        """
        super().__init__()
        self.threshold = threshold
        self.pE = np.ones((2,)) / 2

    def _update(self, probsA: np.ndarray, probsB: np.ndarray):
        # Get likelihood p(A|E), p(B|E)
        pAE = probsA
        pBE = probsB
        
        pAE_full = np.stack([1-pAE, pAE], axis=1)
        pBE_full = np.stack([1-pBE, pBE], axis=1)

        # Find joint p(A|E)p(B|E)p(E) and marginal
        joint = pAE_full * pBE_full * self.pE
        marginal = np.sum(joint, axis=1)

        # Find posterior
        posterior = joint / marginal
        return posterior

    def getFusedDecision(self, probsA: np.ndarray, probsB: np.ndarray) -> bool:
        posterior = self._update(probsA, probsB)

        # p(E=1 | A=1, B=1)
        if posterior[1,1] > self.threshold:
            return True
        else:
            return False

# ------------------ Decision Tree ------------------ #
def doDecision(
    ev_detected: bool,
    audio_signal: np.ndarray,
    bayesFilter: PassageBayesFilter
):
    if ev_detected:

        if not bayesFilter.hasPassed(audio_signal):
           return Decision.STOP
        else:
            return Decision.GO
    
    else:
        return Decision.GO
    
# ------------------ Test Cases ------------------ #
def test_infoFusion(outputA, outputB, is_ev_detected):
    expected_classifications = {
        (True, True): True,
        (False, False): False,
        (True, False): False, # Simplified
        (False, True): True   # Simplified, Assumption
    }

    threshold = 0.7
    b_bool = bool((outputB[1] > threshold).item())
    expected = expected_classifications[(outputA, b_bool)]

    if expected == is_ev_detected:
        color = "\033[92m"   # green
    else:
        color = "\033[91m"   # red

    reset = "\033[0m"

    logger.collect(f"{color}Should be {expected}, got {is_ev_detected}{reset}")

class SystemEvaluation(object):
    def __init__(self):
        super().__init__()
        self.raw_folder = Path('./data/raw/')

        self.raw_audios = list(self.raw_folder.glob('*.wav'))
        self.raw_lights = list(self.raw_folder.glob('*.csv'))

        self.test_cases: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.metrics: List[Dict[str, Any]] = []

    def test(self, f: Callable, *args):
        labels, cases = self._build_test_cases()

        # Labels: mic_true_nopass_{i}.wav
        # Separate by _

        for audio, light, loops in cases:
            print(labels, cases)
            decision = f(audio, light, loops, *args)
            # Collect results

        # Confusion matrix
        # 

        return self.metrics
    
    def _load_light_csv(self, path: Path) -> Tuple[Path, np.ndarray, np.ndarray]:
        """ Loads CSV containing light + loop time columns. """
        filename = path.name

        light_vals = []
        loop_times = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                light_vals.append(float(row["light"]))
                loop_times.append(float(row["loop_time"]))

        return filename, np.array(light_vals), np.array(loop_times)
    
    def _load_audio_waveforms(self, path: Path, sr: int=8000) -> Tuple[Path, np.ndarray]:
        filename = path.name
        audio, sr = lr.load(path, sr=sr)
        return filename, audio
    
    def _build_test_cases(self) -> Tuple[List[str], List[Tuple]]:
        labels: List[str] = []
        cases: List[Tuple] = []

        # Load everything from raw_audio and raw_light
        audio_cases = [self._load_audio_waveforms(p) for p in self.raw_audios]
        light_cases = [self._load_light_csv(p) for p in self.raw_lights]

        # Build maps by filename stem
        audio_map = {p.stem: (p, a) for p, a in audio_cases}
        light_map = {p.stem: (p, l, loops) for p, l, loops in light_cases}

        common_keys = sorted(audio_map.keys() & light_map.keys())

        max_window_sec = 1.5

        # (Comment here)
        for key in common_keys:
            _, audio = audio_map[key]
            _, light, loops = light_map[key]

            n = len(light)
            start = 0

            # Time-based chunking
            while start < n:
                acc_time = 0.0
                end = start

                while end < n and acc_time < max_window_sec:
                    acc_time += float(loops[end])
                    end += 1

                # Skip useless small chunks
                if end - start < 2:
                    break

                light_chunk = light[start:end]
                loop_chunk = loops[start:end]

                labels.append(key)
                cases.append((audio, light_chunk, loop_chunk))

                start = end

        return labels, cases

# ------------------ Recognition / Main Logic ------------------ #
def recognize(
    audio: np.ndarray,
    light: np.ndarray,
    loops: np.ndarray,
    light_model: EmergencyVehicleLightModel,
    audio_model: Tuple[WaveformResNet, float],
    info_fusion: InformationFusion,
    bayesFilter: PassageBayesFilter,
    n_fft: float,
) -> Decision:
    # *** Data processing *** #
    # light_processed = normalizeSignal(light)
    # audio_processed = normalizeSignal(audio)

    light_processed = light
    audio_processed = filterBandpass(audio)

    # *** Fourier Transform *** #
    audio_processed = performSTFT(audio_processed, sr=mic.fs, n_fft=n_fft)
    # light_processed = performFFT(light_processed)

    # *** Outlier detection ***
    # audio_outliers = detectOutliers(audio_processed)
    light_outliers = detectOutliers(light_processed)

    # *** Sensor Fusion *** # 
    #     Sensor fusion between audio and visual frequency
    #     for increased confidence for state estimation
    #     but if light stationary, while audio is dynamic
    #     this would cause issues. Skipped.

    # *** Classification *** #
    # Input:  [lightBuffer]
    # Output: [bool per class]
    outputA = inferLight(light_processed, loops, light_model, light_outliers)
    # Input:  [audioBuffer]
    # Output: [probs per class]
    outputB = inferSound(audio_processed, *audio_model)
    # Input:  [Heuristic output, ML output]
    # Output: EV detected or EV not detected
    is_ev_detected = info_fusion.getFusedDecision(booleanToProbs(outputA), outputB)

    # For testing purposes, comment if not needed
    b_bool = bool((outputB[1] > 0.7).item())
    logger.collect(f"Light: {outputA}, Sound: {b_bool}")
    bayesFilter.hasPassed(audio)
    
    # *** Decision Making *** #
    decision = doDecision(is_ev_detected, audio, bayesFilter)
    return decision

# ------------------ Core ------------------ #

if __name__ == '__main__':
    # Initialize logger
    logger = MsgLogger()
    
    # Note: Change port and device accordingly
    # Note: Device=2 when headphone is used
    device = 2
    port = '/dev/tty.usbserial-D30HY4NS'

    try:
        ser = serial.Serial(
            port=port,
            baudrate=115200,
            timeout=0.5,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
    except serial.SerialException:
        ser = None
        logger.log('\033[91m'"No serial device found."'\033[0m')

    # Initialize sensors
    mic = MicInterface(device)
    board = SensorInterface(ser)
    board_thread = SensorThread(board)

    # Start collecting from sensors. Separate threads per process.
    mic.start()
    board_thread.start()

    # Initialize variables
    n_fft = 512
    n_bins = 247
    n_mels = 128
    light_model = EmergencyVehicleLightModel()
    info_fusion = InformationFusion()
    bayesFilter = PassageBayesFilter()

    # Initialize audio classification model
    # mel-spectrogram, duration=4, sr=8000, n_fft=512, n_bins=247, n_mels=128, hop_length=128
    model_path = "waveform_model_e2.pth"
    temperature = 1.6207691431045532

    model = WaveformResNet(
        bin_shape=n_bins, 
        mel_shape=n_mels,
        hidden_channels=48,
        n_resBlocks=3,
        gru_hidden=48,
        gru_layers=3, 
        attn_hidden=32,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    audio_model = (model, temperature)

    # Warmup sensors here
    time.sleep(4)
    logger.log('\033[92m'"Initialized"'\033[0m')

    try:
        while True:
            audio = mic.get_audio()
            light = np.array(board.get_measurements("light_volt"))
            loops = np.array(board.get_measurements("loop_time"))

            decision = recognize(
                audio, light, loops, 
                light_model, audio_model, info_fusion, bayesFilter,
                n_fft
            )

            logger.collect(f"\033[92m{decision}\033[0m") 
            logger.logAll()
            time.sleep(1)
    
    except KeyboardInterrupt:
        mic.stop()
        board_thread.stop()
        ser.close()

    finally:
        mic.stop()
        board_thread.stop()
        ser.close()