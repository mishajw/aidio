import logging
import os

import numpy as np
import tensorflow as tf

from aidio import AudioAutoencoder, normalize_audio, slice_audio, unslice_audio, \
    unnormalize_audio, write_audio

LOG = logging.getLogger(__name__)


def create_experiments(
        audio_autoencoder: AudioAutoencoder,
        audio: np.array,
        audio_path: str,
        encoded_size: int,
        slice_size: int,
        output_directory: str,
        session: tf.Session) -> None:
    LOG.info(f"Creating experiments in {output_directory}")

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    def run_with_modifier(modifier: np.array, output_file: str):
        normalized, std = normalize_audio(audio)
        sliced = slice_audio(
            [normalized], slice_size, preserve_order=True)
        decoded_sliced = session.run(
            audio_autoencoder.audio_output,
            {
                audio_autoencoder.audio_input: sliced,
                audio_autoencoder.encoded_modifier: modifier
            })
        unsliced = unslice_audio(decoded_sliced)
        unnormalized = unnormalize_audio(unsliced, std)
        write_audio(
            unnormalized,
            output_file,
            audio_path)

    for i in range(encoded_size):
        modifier = np.ones(encoded_size)
        modifier[i] = 1.5
        run_with_modifier(
            modifier, os.path.join(output_directory, f"experiment_{i}_up.wav"))

        modifier[i] = 0.5
        run_with_modifier(
            modifier,
            os.path.join(output_directory, f"experiment_{i}_down.wav"))
