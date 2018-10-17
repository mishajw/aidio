import numpy as np
from typing import List

from aidio import AudioAutoencoder, read_audio


def train(training_files: List[str], slice_size: int, encoded_size: int):
    audio = list(map(read_audio, training_files))
    audio_slices = __get_slices(audio, slice_size)

    audio_autoencoder = AudioAutoencoder(
        slice_size, encoded_size, [4, 4, 4], 256)
    # TODO: Train


def __get_slices(audio: List[np.array], slice_size: int) -> np.array:
    # Remove any remainders when audio length isn't multiple of `slice_size`
    trimmed = [
        a[:len(a) // slice_size * len(a)]
        for a in audio]
    sliced = [
        np.reshape(a, (-1, slice_size))
        for a in audio]
    return np.concatenate(sliced)
