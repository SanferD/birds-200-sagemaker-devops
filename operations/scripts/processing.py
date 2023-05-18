import argparse
import logging
import os
import pathlib
import random
import shutil
import tarfile
import urllib.request
from typing import Tuple

import convert_recordio
import numpy as np
import pandas as pd
import PIL.Image
import shared_constants
import sklearn.model_selection


logging.basicConfig(level=logging.INFO)


# Constants for dataset
BIRDS_200_DIR = "CUB_200_2011"
BIRDS_200_DATASET_URL = (
    f"https://s3.amazonaws.com/fast-ai-imageclas/{BIRDS_200_DIR}.tgz"
)
IM2REC_GITHUB_URL = (
    "https://raw.githubusercontent.com/apache/mxnet/master/tools/im2rec.py"
)

# Column names for dataframes
IMAGE_NAME_COL = "image_name"
IMAGE_ID_COL = "image_id"
CLASS_ID_COL = "class_id"
X_ABS_COL = "x_abs"
Y_ABS_COL = "y_abs"
BBOX_WIDTH_COL = "bbox_width"
BBOX_HEIGHT_COL = "bbox_height"
IMAGE_HEIGHT_COL = "image_height"
IMAGE_WIDTH_COL = "image_width"
BOX_X0_COL = "box_x0"
BOX_X1_COL = "box_x1"
BOX_Y0_COL = "box_y0"
BOX_Y1_COL = "box_y1"
OBJECT_WIDTH_COL = "object_width"
EXTRA_HEADER_WIDTH_COL = "extra_header_width"
IS_TRAIN_COL = "is_train"

# Column names for LST file
LST_COLS = [
    EXTRA_HEADER_WIDTH_COL,
    OBJECT_WIDTH_COL,
    CLASS_ID_COL,
    BOX_X0_COL,
    BOX_Y0_COL,
    BOX_X1_COL,
    BOX_Y1_COL,
    IMAGE_NAME_COL
]

# Paths to data files and directories
BIRDS_200_PATH = shared_constants.PROCESSING_DIR / BIRDS_200_DIR
CLASSES_TXT_PATH = BIRDS_200_PATH / "classes.txt"
BOUNDING_BOX_TXT_PATH = BIRDS_200_PATH / "bounding_boxes.txt"
IMAGES_TXT_PATH = BIRDS_200_PATH / "images.txt"
IMAGE_CLASS_LABELS_TXT_PATH = BIRDS_200_PATH / "image_class_labels.txt"
SIZES_TXT_PATH = BIRDS_200_PATH / "sizes.txt"
TRAIN_TEST_SPLIT_TXT_PATH = BIRDS_200_PATH / "train_test_split.txt"
IMAGES_DIR_PATH = BIRDS_200_PATH / "images"

# Split fractions for train, validation and test sets
TRAIN_FRAC = 0.70
TEST_FRAC = 0.15
VAL_FRAC = 0.15

# LST file names for train, validation and test sets
TRAIN_LST_FILE_NAME = (
    f"{shared_constants.PACKAGE_LABEL}_{shared_constants.TRAIN_CHANNEL}.lst"
)
VAL_LST_FILE_NAME = (
    f"{shared_constants.PACKAGE_LABEL}_{shared_constants.VALIDATION_CHANNEL}.lst"
)
TEST_LST_FILE_NAME = (
    f"{shared_constants.PACKAGE_LABEL}_{shared_constants.TEST_CHANNEL}.lst"
)


def main():
    create_directories()
    dataset_path = download_file(
        BIRDS_200_DATASET_URL,
        download_dir=shared_constants.PROCESSING_DIR
    )
    if not BIRDS_200_PATH.exists():
        extract_dataset(dataset_path)
    full_df = merge_datasets()
    full_df = add_image_size_to_df(full_df)
    add_relative_box_coordinates(full_df)
    transform_to_zero_based_class_ids(full_df)
    add_lst_headers(full_df)
    train_df, val_df, test_df = get_lst_train_val_test_split(full_df)
    logging.info(
        f"len(train_df)={len(train_df)}, "
        f"len(val_df)={len(val_df)}, "
        f"len(test_df)={len(test_df)}"
    )
    assert len(train_df) == shared_constants.NUM_TRAINING_SAMPLES, \
        "len(train_df) != #train-samples"
    generate_record_io_files()
    move_train_val_recordio_files()
    extract_test_images_and_move_to_test_dir()
    
    
def create_directories():
    """Create necessary directories."""
    logging.info("Creating directories")
    shared_constants.PROCESSING_DIR.mkdir(exist_ok=True)
    shared_constants.TRAIN_DIR.mkdir(exist_ok=True)
    shared_constants.VALIDATION_DIR.mkdir(exist_ok=True)
    shared_constants.TEST_DIR.mkdir(exist_ok=True)
    shared_constants.LABELS_DIR.mkdir(exist_ok=True)


def extract_dataset(dataset_path: str):
    """Extract dataset from compressed file."""
    logging.info(f"Extracting dataset from {dataset_path}")
    compressed = tarfile.open(dataset_path)
    compressed.extractall(shared_constants.PROCESSING_DIR)


def merge_datasets() -> pd.DataFrame:
    """Merge datasets from different sources."""
    logging.info("Merging datasets")

    # Read data from files
    classes_df = pd.read_csv(
        CLASSES_TXT_PATH,
        sep=" ",
        names=[CLASS_ID_COL, "class_name"],
        header=None
    )
    bounding_box_df = pd.read_csv(
        BOUNDING_BOX_TXT_PATH,
        sep=" ",
        names=[
            IMAGE_ID_COL,
            X_ABS_COL,
            Y_ABS_COL,
            BBOX_WIDTH_COL,
            BBOX_HEIGHT_COL
        ],
        header=None
    )
    images_df = pd.read_csv(
        IMAGES_TXT_PATH,
        sep=" ",
        names=[IMAGE_ID_COL, IMAGE_NAME_COL],
        header=None
    )
    image_class_labels_df = pd.read_csv(
        IMAGE_CLASS_LABELS_TXT_PATH,
        sep=" ",
        names=[IMAGE_ID_COL, CLASS_ID_COL],
        header=None
    )
    train_test_split_df = pd.read_csv(
        TRAIN_TEST_SPLIT_TXT_PATH,
        sep=" ",
        names=[IMAGE_ID_COL, IS_TRAIN_COL],
        header=None
    )

    # Merge dataframes
    full_df = bounding_box_df.merge(images_df, how="outer", on=IMAGE_ID_COL)
    full_df = full_df.merge(image_class_labels_df, how="outer", on=IMAGE_ID_COL)
    full_df = full_df.merge(train_test_split_df, how="outer", on=IMAGE_ID_COL)
    full_df = full_df.merge(classes_df, how="outer", on=CLASS_ID_COL)

    # Filter by class IDs if specified
    if shared_constants.CLASS_IDS:
        full_df = full_df[full_df[CLASS_ID_COL].isin(shared_constants.CLASS_IDS)]

    return full_df


def add_image_size_to_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Add image width and height to dataframe."""
    logging.info("Adding image width and height to dataframe")
    extra_entries = []
    for i, row in full_df.iterrows():
        img_path = IMAGES_DIR_PATH / row[IMAGE_NAME_COL]
        img = PIL.Image.open(str(img_path))
        width, height = img.size
        entry = {
            IMAGE_HEIGHT_COL: height,
            IMAGE_WIDTH_COL: width,
            IMAGE_ID_COL: row[IMAGE_ID_COL],
        }
        extra_entries.append(entry)

    full_df = pd.DataFrame(extra_entries).merge(
        full_df,
        how="outer",
        on=IMAGE_ID_COL
    )
    return full_df


def add_relative_box_coordinates(full_df: pd.DataFrame):
    """Add relative box coordinates to dataframe."""
    logging.info("Adding relative box coordinates to dataframe")
    full_df[BOX_X0_COL] = full_df[X_ABS_COL] / full_df[IMAGE_WIDTH_COL]
    full_df[BOX_X1_COL] = (
        full_df[X_ABS_COL] + full_df[BBOX_WIDTH_COL]
    ) / full_df[IMAGE_WIDTH_COL]
    full_df[BOX_Y0_COL] = full_df[Y_ABS_COL] / full_df[IMAGE_HEIGHT_COL]
    full_df[BOX_Y1_COL] = (
        full_df[Y_ABS_COL] + full_df[BBOX_HEIGHT_COL]
    ) / full_df[IMAGE_HEIGHT_COL]


def transform_to_zero_based_class_ids(full_df: pd.DataFrame):
    """Transform class IDs to zero-based."""
    logging.info("Transforming class IDs to zero-based")
    class_id_to_new_class_id = dict(
        (k, v) for (v, k) in enumerate(shared_constants.CLASS_IDS)
    )
    full_df[CLASS_ID_COL] = (
        full_df[CLASS_ID_COL].map(class_id_to_new_class_id).astype(float)
    )


def add_lst_headers(full_df: pd.DataFrame) -> pd.DataFrame:
    """Add headers to LST dataframe."""
    logging.info("Adding headers to LST dataframe")
    full_df[EXTRA_HEADER_WIDTH_COL] = 2
    full_df[OBJECT_WIDTH_COL] = 5
    lst_df = full_df.reset_index()
    return lst_df

    
def get_lst_train_val_test_split(
    full_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, validation and test sets and save as LST files."""
    logging.info("Splitting dataframe into train, validation and test sets")
    train_df, val_df, test_df = _train_val_test_split(full_df)

    # Save dataframes as LST files
    train_df[LST_COLS].to_csv(
        TRAIN_LST_FILE_NAME,
        sep="\t",
        float_format="%.4f",
        header=False
    )
    val_df[LST_COLS].to_csv(
        VAL_LST_FILE_NAME,
        sep="\t",
        float_format="%.4f",
        header=False
    )
    test_df[LST_COLS].to_csv(
        TEST_LST_FILE_NAME,
        sep="\t",
        float_format="%.4f",
        header=False
    )

    return train_df, val_df, test_df


def _train_val_test_split(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, validation and test sets."""
    assert TRAIN_FRAC + VAL_FRAC + TEST_FRAC == 1.0, \
        "Train fraction + validation fraction + test fraction != 1.0"

    # Calculate number of samples for each set
    n_samples = len(df)
    n_train = int(n_samples * TRAIN_FRAC)
    n_val = int(n_samples * VAL_FRAC)
    n_test = int(n_samples * TEST_FRAC)
    n_total = n_train + n_val + n_test
    n_train += n_samples - n_total
    assert n_train + n_val + n_test == n_samples, (
        "Number of train samples + number of validation samples "
        "+ number of test samples != number of samples"
    )

    # Split dataframe into train, validation and test sets
    train_rows = []
    val_rows = []
    test_rows = []
    key2rows = {
        shared_constants.TRAIN_CHANNEL: train_rows,
        shared_constants.VALIDATION_CHANNEL: val_rows,
        shared_constants.TEST_CHANNEL: test_rows
    }
    key2n = {
        shared_constants.TRAIN_CHANNEL: n_train,
        shared_constants.VALIDATION_CHANNEL: n_val,
        shared_constants.TEST_CHANNEL: n_test
    }
    keys = list(key2rows.keys())
    for _, row in df.iterrows():
        assert len(keys) > 0, "len(keys) > 0 is violated"
        key = random.choice(keys)
        key2rows[key].append(row)
        if len(key2rows[key]) == key2n[key]:
            keys.remove(key)
    return (
        pd.DataFrame(train_rows),
        pd.DataFrame(val_rows),
        pd.DataFrame(test_rows)
    )


def generate_record_io_files():
    """Generate RecordIO files."""
    logging.info("Generating RecordIO files")
    download_file(
        IM2REC_GITHUB_URL,
        download_dir=pathlib.Path(".")
    )
    command = (
        f"python3 im2rec.py --resize {shared_constants.RESIZE_SIZE} "
        f"--pack-label {shared_constants.PACKAGE_LABEL} "
        f"{IMAGES_DIR_PATH}"
    )
    logging.info(command)
    os.system(command)


def download_file(
    url: str,
    download_dir: pathlib.Path,
    force: bool = False
) -> pathlib.Path:
    """Download file from URL."""
    filename = url.split("/")[-1]
    filepath = download_dir / filename
    if force or not filepath.exists():
        urllib.request.urlretrieve(url, filepath)
    return filepath


def move_train_val_recordio_files():
    """Move train and validation RecordIO files to their respective directories."""
    logging.info(
        "Moving train and validation RecordIO files to their respective directories"
    )
    for rec_file, dst in [
        (shared_constants.TRAIN_RECORD_FILE_NAME, shared_constants.TRAIN_DIR),
        (shared_constants.VALIDATION_RECORD_FILE_NAME, shared_constants.VALIDATION_DIR),
    ]:
        logging.info(f"Moving {rec_file} to {dst}")
        shutil.move(rec_file, str(dst))


def extract_test_images_and_move_to_test_dir():
    """Extract test images from RecordIO file and move them to test directory."""
    logging.info(
        "Extracting test images from RecordIO file and moving them to test directory"
    )
    convert_recordio.convert_recordio(
        shared_constants.TEST_RECORD_FILE_NAME,
        shared_constants.TEST_DIR,
        shared_constants.LABELS_DIR
    )
        
    
if __name__ == "__main__":
    main()
