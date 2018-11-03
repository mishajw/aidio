import logging
import os
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np

import tf_utils
from aidio import AudioAutoencoder, read_audio, write_audio, slice_audio, \
    unslice_audio, normalize_audio, create_experiments

LOG = logging.getLogger(__name__)
BATCH_SIZE = 2


def train(
        training_files: List[str],
        slice_size: int,
        encoded_size: int,
        output_directory: str):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    LOG.info("Reading training data")
    audio = list(map(read_audio, training_files))
    # Take the first audio example and set it aside for testing
    test_audio, *audio = audio
    audio, stds = zip(*[normalize_audio(a) for a in audio])
    audio_slices = slice_audio(audio, slice_size)

    LOG.info("Creating model")
    audio_autoencoder = AudioAutoencoder(slice_size, encoded_size, [4, 4, 4])

    LOG.info("Setting up training")
    data_holder = tf_utils.data_holder.DataHolder.from_input_output_lists(
        args=SimpleNamespace(cap_data=None, testing_percentage=20),
        input_list=audio_slices,
        output_list=[0] * len(audio_slices))
    runner = tf_utils.generic_runner.GenericRunner(
        "aidio", training_steps=None, testing_step=100, batch_size=BATCH_SIZE,
        add_all_summaries=False, run_tag=None)
    runner.set_data_holder(data_holder)
    runner.set_get_feed_dict(lambda t: {
        audio_autoencoder.audio_input: t[0],
        audio_autoencoder.encoded_modifier: np.ones(encoded_size)
    })
    runner.set_train_evaluations([audio_autoencoder.optimizer])
    runner.set_test_evaluations([audio_autoencoder.loss])

    num_test_calls = 0

    def test_callback(values):
        nonlocal num_test_calls
        print(num_test_calls, "Test score: ", values)
        if num_test_calls % 10 == 0 and num_test_calls != 0:
            create_experiments(
                audio_autoencoder, test_audio, training_files[0], encoded_size,
                slice_size,
                os.path.join(output_directory, f"experiments_{num_test_calls}"),
                runner.get_session())
        num_test_calls += 1

    runner.set_test_callback(test_callback)

    LOG.info("Training")
    runner.run()
