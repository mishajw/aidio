import argparse
import logging

from aidio import train


def main():
    parser = argparse.ArgumentParser("aidio")
    parser.add_argument("--training-files", nargs="+", type=str, required=True)
    parser.add_argument("--slice-size", type=int, default=256)
    parser.add_argument("--encoded-size", type=int, default=16)
    parser.add_argument("--output-directory", type=str, default="output")
    args = parser.parse_args()
    train(
        args.training_files,
        args.slice_size,
        args.encoded_size,
        args.output_directory)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
