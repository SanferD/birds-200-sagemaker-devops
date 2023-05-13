import os
import pathlib
import urllib.request
import tarfile
import pandas as pd
import sklearn.model_selection
import random
import argparse
import numpy as np
import shared_constants
import PIL.Image
import shutil
import logging


logging.basicConfig(level=logging.INFO)


BIRDS_200_DIR = "CUB_200_2011"
BIRDS_200_DATASET_URL = f"https://s3.amazonaws.com/fast-ai-imageclas/{BIRDS_200_DIR}.tgz"
IM2REC_GITHUB_URL = "https://raw.githubusercontent.com/apache/mxnet/master/tools/im2rec.py"

IMAGE_NAME = "image_name"
IMAGE_ID = "image_id"
CLASS_ID = "class_id"
X_ABS = "x_abs"
Y_ABS = "y_abs"
BBOX_WIDTH = "bbox_width"
BBOX_HEIGHT = "bbox_height"
IMAGE_NAME = "image_name"
IMAGE_HEIGHT = "image_height"
IMAGE_WIDTH = "image_width"
BOX_X0 = "box_x0"
BOX_X1 = "box_x1"
BOX_Y0 = "box_y0"
BOX_Y1 = "box_y1"
OBJECT_WIDTH = "object_width"
EXTRA_HEADER_WIDTH = "extra_header_width"
IS_TRAIN = "is_train"

LST_COLS = [EXTRA_HEADER_WIDTH, OBJECT_WIDTH, CLASS_ID, BOX_X0, BOX_Y0, BOX_X1, BOX_Y1, IMAGE_NAME]

ML_DATA_BIRDS_200_DIR = shared_constants.ML_PROC_DIR / BIRDS_200_DIR
CLASSES_TXT = ML_DATA_BIRDS_200_DIR / "classes.txt"
BOUNDING_BOX_TXT = ML_DATA_BIRDS_200_DIR / "bounding_boxes.txt"
IMAGES_TXT = ML_DATA_BIRDS_200_DIR / "images.txt"
IMAGE_CLASS_LABELS_TXT = ML_DATA_BIRDS_200_DIR / "image_class_labels.txt"
SIZES_TXT = ML_DATA_BIRDS_200_DIR / "sizes.txt"
TRAIN_TEST_SPLIT_TXT = ML_DATA_BIRDS_200_DIR / "train_test_split.txt"
IMAGES_DIR = ML_DATA_BIRDS_200_DIR / "images"

TRAIN_FRAC = 0.70
TEST_FRAC = 0.15
VAL_FRAC = 0.15

TRAIN_LST = f"{shared_constants.PACK_LABEL}_{shared_constants.TRAIN_CHANNEL}.lst"
VAL_LST = f"{shared_constants.PACK_LABEL}_{shared_constants.VAL_CHANNEL}.lst"
TEST_LST = f"{shared_constants.PACK_LABEL}_{shared_constants.TEST_CHANNEL}.lst"


def main():
    mkdirs()
    dataset_path = _download(BIRDS_200_DATASET_URL, download_dir=shared_constants.ML_PROC_DIR)
    if not ML_DATA_BIRDS_200_DIR.exists():
        extract_dataset(dataset_path)
    full_df = merge_datasets()
    full_df = add_image_width_and_height_to_df(full_df)
    add_relative_box_coordinates(full_df)
    transform_to_zero_based_class_ids(full_df)
    add_lst_headers(full_df)
    (train_df, val_df, test_df) = get_lst_train_val_test_split(full_df)
    logging.info(f"len(train_df)={len(train_df)}, len(val_df)={len(val_df)}, len(test_df)={len(test_df)}")
    assert (len(train_df)==shared_constants.NUM_TRAINING_SAMPLES), "len(train_df) != #train-samples"
    generate_record_io_files()
    upload_record_io_files()
    
    
def mkdirs():
    logging.info("mkdirs")
    shared_constants.ML_DIR.mkdir(exist_ok=True)
    shared_constants.ML_PROC_DIR.mkdir(exist_ok=True)
    shared_constants.ML_TRAIN_DIR.mkdir(exist_ok=True)
    shared_constants.ML_VAL_DIR.mkdir(exist_ok=True)
    shared_constants.ML_TEST_DIR.mkdir(exist_ok=True)


def extract_dataset(dataset_path):
    logging.info(f"extract_dataset {dataset_path}")
    compressed = tarfile.open(dataset_path)
    compressed.extractall(shared_constants.ML_PROC_DIR)
    
    
def merge_datasets():
    logging.info("merge_datasets")
    classes_df = pd.read_csv(CLASSES_TXT, sep=" ", names=[CLASS_ID, "class_name"], header=None)
    bounding_box_df = pd.read_csv(BOUNDING_BOX_TXT, sep=" ", names=[IMAGE_ID, X_ABS, Y_ABS, BBOX_WIDTH, BBOX_HEIGHT], header=None)
    images_df = pd.read_csv(IMAGES_TXT, sep=" ", names=[IMAGE_ID, IMAGE_NAME], header=None)
    image_class_labels_df = pd.read_csv(IMAGE_CLASS_LABELS_TXT, sep=" ", names=[IMAGE_ID, CLASS_ID], header=None)
    train_test_split_df = pd.read_csv(TRAIN_TEST_SPLIT_TXT, sep=" ", names=[IMAGE_ID, IS_TRAIN], header=None)

    full_df = bounding_box_df.merge(images_df, how="outer", on=IMAGE_ID)
    full_df = full_df.merge(image_class_labels_df, how="outer", on=IMAGE_ID)
    full_df = full_df.merge(train_test_split_df, how="outer", on=IMAGE_ID)
    full_df = full_df.merge(classes_df, how="outer", on=CLASS_ID)
    if shared_constants.CLASS_IDS:
        full_df = full_df[  full_df[CLASS_ID].isin(shared_constants.CLASS_IDS)  ]
    return full_df


def add_image_width_and_height_to_df(full_df):
    logging.info("add_image_width_and_height_to_df")
    extra_entries = []
    for i, row in full_df.iterrows():
        img_path = IMAGES_DIR / row[IMAGE_NAME]
        img = PIL.Image.open(str(img_path))
        (width, height) = img.size
        entry = {
            IMAGE_HEIGHT: height,
            IMAGE_WIDTH: width,
            IMAGE_ID: row[IMAGE_ID],
        }
        extra_entries.append(entry)

    full_df = pd.DataFrame(extra_entries).merge(full_df, how="outer", on=IMAGE_ID)
    return full_df


def add_relative_box_coordinates(full_df):
    logging.info("add_relative_box_coordinates")
    full_df[BOX_X0] = full_df[X_ABS] / full_df[IMAGE_WIDTH]
    full_df[BOX_X1] = (full_df[X_ABS] + full_df[BBOX_WIDTH]) / full_df[IMAGE_WIDTH]
    full_df[BOX_Y0] = full_df[Y_ABS] / full_df[IMAGE_HEIGHT]
    full_df[BOX_Y1] = (full_df[Y_ABS] + full_df[BBOX_HEIGHT]) / full_df[IMAGE_HEIGHT]


def transform_to_zero_based_class_ids(full_df):
    logging.info("transform_to_zero_based_class_ids")
    class_id_to_new_class_id = dict((k, v) for (v, k) in enumerate(shared_constants.CLASS_IDS))
    full_df[CLASS_ID] = full_df[CLASS_ID].map(class_id_to_new_class_id).astype(float)


def add_lst_headers(full_df):
    logging.info("add_lst_headers")
    full_df[EXTRA_HEADER_WIDTH] = 2
    full_df[OBJECT_WIDTH] = 5
    lst_df = full_df.reset_index()
    return lst_df

    
def get_lst_train_val_test_split(full_df):
    logging.info("get_lst_train_val_test_split")
    (train_df, val_df, test_df) = _train_val_test_split(full_df)
    
    train_df[LST_COLS].to_csv(TRAIN_LST, sep="\t", float_format="%.4f", header=False)
    val_df[LST_COLS].to_csv(VAL_LST, sep="\t", float_format="%.4f", header=False)
    test_df[LST_COLS].to_csv(TEST_LST, sep="\t", float_format="%.4f", header=False)
    
    return (train_df, val_df, test_df)


def _train_val_test_split(df):
    assert (TRAIN_FRAC + TEST_FRAC + VAL_FRAC == 1.0), "TRAIN_FRAC + VAL_FRAC + TEST_FRAC != 1.0"
    
    l_df = len(df)
    (n_train, n_val, n_test) = (int(l_df*TRAIN_FRAC), int(l_df*VAL_FRAC), int(l_df*TEST_FRAC))
    N = n_train + n_val + n_test
    n_train += l_df - N
    assert (n_train + n_val + n_test == l_df), "n_train + n_val + n_test != l_df"
    
    (train_rows, val_rows, test_rows) = ([], [], [])
    key2rows = {shared_constants.TRAIN_CHANNEL: train_rows, shared_constants.VAL_CHANNEL: val_rows, shared_constants.TEST_CHANNEL: test_rows}
    key2n = {shared_constants.TRAIN_CHANNEL: n_train, shared_constants.VAL_CHANNEL: n_val, shared_constants.TEST_CHANNEL: n_test}
    keys = list(key2rows.keys())
    for _, row in df.iterrows():
        assert len(keys) > 0, "len(keys) > 0 is violated"
        key = random.choice(keys)
        key2rows[key].append(row)
        if len(key2rows[key]) == key2n[key]:
            keys.remove(key)
    return (pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows))


def generate_record_io_files():
    logging.info("generate_record_io_files")
    _download(IM2REC_GITHUB_URL, download_dir=pathlib.Path("."))
    logging.info(f"python3 im2rec.py --resize {shared_constants.RESIZE_SIZE} --pack-label {shared_constants.PACK_LABEL} {IMAGES_DIR}")
    os.system(f"python3 im2rec.py --resize {shared_constants.RESIZE_SIZE} --pack-label {shared_constants.PACK_LABEL} {IMAGES_DIR}")


def _download(url, download_dir, force=False):
    filename = url.split("/")[-1]
    filepath = download_dir / filename
    if force or not filepath.exists():
        urllib.request.urlretrieve(url, filepath)
    return filepath


def upload_record_io_files():
    logging.info("upload_record_io_files")
    for (rec_file, dst) in [(shared_constants.TRAIN_REC, shared_constants.ML_TRAIN_DIR),
                               (shared_constants.VAL_REC, shared_constants.ML_VAL_DIR),
                               (shared_constants.TEST_REC, shared_constants.ML_TEST_DIR)]:
        logging.info(f"mv {rec_file} {dst}")
        shutil.move(rec_file, str(dst))
    
    
if __name__ == "__main__":
    main()
