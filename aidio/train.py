import logging
from types import SimpleNamespace
from typing import List

import numpy as np

import tf_utils
from aidio import AudioAutoencoder, read_audio

LOG = logging.getLogger(__name__)
BATCH_SIZE = 2


def train(training_files: List[str], slice_size: int, encoded_size: int):
    LOG.info("Reading training data")
    audio = list(map(read_audio, training_files))
    audio_slices = __get_slices(audio, slice_size)

    LOG.info("Creating model")
    audio_autoencoder = AudioAutoencoder(
        slice_size, encoded_size, [4, 4, 4], BATCH_SIZE)

    LOG.info("Setting up training")
    data_holder = tf_utils.data_holder.DataHolder.from_input_output_lists(
        args=SimpleNamespace(cap_data=None, testing_percentage=20),
        input_list=audio_slices,
        output_list=[0] * len(audio_slices))
    runner = tf_utils.generic_runner.GenericRunner(
        "aidio", training_steps=None, testing_step=10, batch_size=BATCH_SIZE,
        add_all_summaries=False, run_tag=None)
    runner.set_data_holder(data_holder)
    runner.set_get_feed_dict(lambda t: {audio_autoencoder.audio_input: t[0]})
    runner.set_train_evaluations([audio_autoencoder.optimizer])
    runner.set_test_evaluations([audio_autoencoder.loss])
    runner.set_test_callback(lambda data: LOG.info("Test results: %s", data))

    LOG.info("Training")
    runner.run()


def __get_slices(audio: List[np.array], slice_size: int) -> np.array:
    # Remove any remainders when audio length isn't multiple of `slice_size`
    trimmed = [
        a[:len(a) // slice_size * len(a)]
        for a in audio]
    sliced = [
        np.reshape(a, (-1, slice_size))
        for a in audio]
    return np.concatenate(sliced)
