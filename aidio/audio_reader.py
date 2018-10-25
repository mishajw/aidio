import wave
from typing import List

import numpy as np

BUFFER_SIZE = 32
MAX_NUM_FRAMES = 1000


def read_audio(path: str) -> np.array:
    pieces: List[np.array] = []
    with wave.open(path, 'r') as f:
        assert f.getsampwidth() == 2
        audio_dtype = np.dtype("int16").newbyteorder("L")

        for _ in range(0, min(MAX_NUM_FRAMES, f.getnframes()), BUFFER_SIZE):
            # Read into numpy array
            bytes = f.readframes(BUFFER_SIZE)
            array = np.frombuffer(bytes, audio_dtype)
            # Convert shape to [num_samples, num_channels]
            array = np.reshape(array, (-1, f.getnchannels()))
            # Average across all channels
            array = np.average(array, 1)
            pieces.append(array)

    return np.concatenate(pieces)
