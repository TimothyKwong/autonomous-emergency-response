from collections import deque

import sounddevice as sd
import soundfile as sf

import numpy as np

class MicInterface:
    def __init__(
        self, 
        device=2,
        fs=8000, 
        channels=1, 
        duration=4, 
        block_size=512
    ):
        """
        Initialize an audio recorder configuration.

        Parameters
        ----------
        device: int, optional
            ...

        fs : int, optional
            Sampling rate in Hz. Determines how many audio samples are
            collected per second. Default is 8000.
        
        channels : int, optional
            Number of audio channels. Typically 1 for mono or 2 for stereo.
            Default is 1.
        
        duration : int or float, optional
            Size of the rolling audio buffer in seconds. Determines how much
            recent audio is kept in memory at any time. Default is 5 seconds.
        
        block_size : int, optional
            Number of audio frames provided to the callback at once. Smaller
            blocks reduce latency but increase callback frequency. Default is 512.
        """
        super().__init__()
        
        self.device = device
        self.fs = fs
        self.channels = channels
        self.duration = duration
        self.block_size = block_size

        self.audio_buffer = deque(maxlen=self.duration * self.fs)
        self.stream = None

    def _callback(self, indata, frames, time, status):
        self.audio_buffer.extend(indata[:, 0].copy())

    def start(self):
        if self.stream is not None:
            return

        self.stream = sd.InputStream(
            samplerate=self.fs,
            channels=self.channels,
            blocksize=self.block_size,
            device=self.device,
            callback=self._callback
        )

        self.stream.start()

    def stop(self):
        """ Stop stream and close. """
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def save(self, filename='mic_audio_test.wav'):
        sf.write(filename, self.get_audio(), self.fs)

    def get_audio(self) -> np.ndarray:
        return np.array(self.audio_buffer, dtype=np.float32)
    
    def get_audio_length(self) -> float:
        # return self.duration
        return len(self.audio_buffer) / self.fs

if __name__ == '__main__':
    audio = MicInterface()
    audio.start()

    try:
        while True:
            pass

    except KeyboardInterrupt:
        audio.stop()
        audio.save()
