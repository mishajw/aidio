import logging
from typing import List, Optional

import tensorflow as tf

LOG = logging.getLogger(__name__)


class AudioAutoencoder:
    def __init__(
            self, input_size: int, encoded_size: int, conv_sizes: List[int]):
        self.input_size = input_size
        self.encoded_size = encoded_size
        self.conv_sizes = conv_sizes
        self.audio_input: Optional[tf.Tensor] = None
        self.audio_encoded: Optional[tf.Tensor] = None
        self.audio_output: Optional[tf.Tensor] = None
        self.encoded_modifier: Optional[tf.Tensor] = None
        self.loss: Optional[tf.Tensor] = None
        self.optimizer: Optional[tf.train.Optimizer] = None
        self.create_model()

    def create_model(self) -> None:

        # The size before and after the fully connected layers around the
        # encoded audio
        pre_encoded_size = self.input_size // 2 ** len(self.conv_sizes)

        self.audio_input = tf.placeholder(
            tf.float32, (None, self.input_size), "audio_input")
        self.encoded_modifier = tf.placeholder(
            tf.float32, self.encoded_size, "encoded_modifier")
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
        audio_encoded_modified = self.audio_encoded * self.encoded_modifier

        # Fully connected into decode input
        decode_input = tf.layers.dense(audio_encoded_modified, pre_encoded_size)
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
                output_shape=tf.stack([
                    tf.shape(decode_input)[0],
                    tf.shape(decode_input)[1] * 2,
                    1]),
                stride=2)
            LOG.debug(
                "Created decoding layer %d, output shape is %s",
                i, decode_input.shape)

        self.audio_output = tf.squeeze(decode_input, 2)
        self.loss = tf.losses.mean_squared_error(
            self.audio_input, self.audio_output)
        tf.summary.scalar("loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
