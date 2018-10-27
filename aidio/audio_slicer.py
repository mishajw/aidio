from typing import List

import numpy as np


def slice_audio(
        audio: List[np.array],
        slice_size: int,
        preserve_order: bool = False) -> np.array:
    # Remove any remainders when audio length isn't multiple of `slice_size`
    trimmed = [
        a[:(len(a) // slice_size) * slice_size]
        for a in audio]
    sliced = [
        np.reshape(a, (-1, slice_size))
        for a in trimmed]
    all_slices = np.concatenate(sliced)
    if not preserve_order:
        all_slices = [
            slice for slice in all_slices
            if np.std(slice) > 1]
        np.random.shuffle(all_slices)
    return all_slices


def unslice_audio(sliced_audio: np.array) -> np.array:
    return np.concatenate(sliced_audio)
