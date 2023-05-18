import argparse
import json
import logging
import os
from typing import Tuple

import cv2
import mxnet as mx
import shared_constants


logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rec-file-path',
        type=str,
        default=shared_constants.BIRDS_200_REC_INPUT_FILE_PATH
    )
    parser.add_argument(
        '--output-data-dir',
        type=str,
        default=shared_constants.ML_OUTPUT_DIR
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    rec_file_path = args.rec_file_path
    output_data_dir = args.output_data_dir
    convert_recordio(rec_file_path, output_data_dir)


def convert_recordio(
    rec_file_path: str,
    output_data_dir: str,
    output_label_dir: str
) -> None:
    """Convert RecordIO file to images and labels.

    Args:
        rec_file_path (str): Path to RecordIO file.
        output_data_dir (str): Path to output data directory.
        output_label_dir (str): Path to output label directory.
    """
    os.makedirs(output_data_dir, exist_ok=True)

    logging.info(f'Reading .rec file from {rec_file_path}')
    record = mx.recordio.MXRecordIO(rec_file_path, 'r')
    logging.info(f'Read .rec file from {rec_file_path}')

    i = 0
    while True:
        item = record.read()
        if not item:
            break
        logging.info(f'Unpacking image {i}')
        header, img = mx.recordio.unpack_img(item)
        logging.info(f'Unpacked image {i}')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_file_path = os.path.join(output_data_dir, f'img_{i}.jpg')

        logging.info(f'Saving image {img_file_path}')
        cv2.imwrite(img_file_path, img)
        logging.info(f'Saved image {img_file_path}')

        label_file_path = os.path.join(output_label_dir, f'label_{i}.json')
        logging.info(f'Saving label {label_file_path}')
        with open(label_file_path, "w+") as label_file:
            json.dump(header.label.tolist(), label_file)
        logging.info(f'Saved label {label_file_path}')

        i += 1


if __name__ == '__main__':
    main()
