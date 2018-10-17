from aidio import AudioAutoencoder


def train():
    audio_autoencoder = AudioAutoencoder(128, 16, [4, 4, 4], 256)
    # TODO: Train
