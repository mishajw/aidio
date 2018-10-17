import logging
from typing import List, Optional

import tensorflow as tf

LOG = logging.getLogger(__name__)


class AudioAutoencoder:
    def __init__(
            self, input_size: int, encoded_size: int, conv_sizes: List[int],
            batch_size: int):
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.conv_sizes = conv_sizes
        # TODO: Remove batch size argument. We should be able to infer this from
        # the input, but it's needed for `tf.contrib.nn.conv1d_transpose`
        self.batch_size = batch_size
        self.audio_input: Optional[tf.Tensor] = None
        self.audio_encoded: Optional[tf.Tensor] = None
        self.create_model()

    def create_model(self) -> None:
        # The size before and after the fully connected layers around the
        # encoded audio
        pre_encoded_size = self.input_size // 2 ** len(self.conv_sizes)

        self.audio_input = tf.placeholder(
            tf.float32, (self.batch_size, self.input_size), "audio_input")
        LOG.debug("Created input with size %s", self.audio_input.shape)

        # Encoding layers, conv1d with stride 2
        encode_input = tf.expand_dims(
            self.audio_input, 2, name="expand_input_dims")
        for i, conv_size in enumerate(self.conv_sizes):
            encode_input = tf.layers.conv1d(
                encode_input, filters=1, kernel_size=conv_size, strides=2,
                padding="same", name=f"encode_conv{i}")
            LOG.debug(
                "Created encoding layer %d, output shape is %s",
                i, encode_input.shape)

        # Fully connected into encoded audio
        encode_flattened = tf.layers.flatten(encode_input)
        LOG.debug(
            "Created flattened layer with shape %s", encode_flattened.shape)
        assert encode_flattened.shape[1] == pre_encoded_size
        encode_fc_output = tf.layers.dense(
            encode_flattened, self.encoded_size)
        self.audio_encoded = encode_fc_output

        # Fully connected into decode input
        decode_input = tf.layers.dense(self.audio_encoded, pre_encoded_size)
        LOG.debug(
            "Created decoder input with shape %s", decode_input.shape)

        # Decoding layers, conv1d with stride 2
        decode_input = tf.expand_dims(
            decode_input, 2, name="expand_decode_dims")
        for i, conv_size in enumerate(reversed(self.conv_sizes)):
            conv_weights = tf.Variable(
                tf.random_uniform((conv_size, 1, 1)),
                name=f"decode_conv{i}_filter")
            decode_input = tf.contrib.nn.conv1d_transpose(
                decode_input,
                conv_weights,
                [self.batch_size, decode_input.shape[1].value * 2, 1],
                stride=2)
            LOG.debug(
                "Created decoding layer %d, output shape is %s",
                i, decode_input.shape)

