import pathlib

# Base dir
BASE_ML_DIR = pathlib.Path("/home/sagemaker-user/opt/ml")

# Define directories
PROCESSING_DIR = BASE_ML_DIR / "processing"
INPUT_DIR = PROCESSING_DIR / "input"
OUTPUT_DIR = PROCESSING_DIR / "output"

# Define channels
TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "val"
TEST_CHANNEL = "test"
LABELS_CHANNEL = "labels"
TRANSFORM_CHANNEL = "transform"
EVALUATION_CHANNEL = "evaluation"

# Define training, validation and test directories
TRAIN_DIR = PROCESSING_DIR / TRAIN_CHANNEL
VALIDATION_DIR = PROCESSING_DIR / VALIDATION_CHANNEL
TEST_DIR = PROCESSING_DIR / TEST_CHANNEL
LABELS_DIR = PROCESSING_DIR / LABELS_CHANNEL
EVALUATION_DIR = PROCESSING_DIR / EVALUATION_CHANNEL
TRANSFORM_DIR = PROCESSING_DIR / TRANSFORM_CHANNEL

# Define S3 bucket and object keys
BUCKET_NAME = "sagemaker-us-east-1-180797159824"
S3_OUTPUT_OBJECT_KEY = f"s3://{BUCKET_NAME}/output"
S3_TRAIN_OBJECT_KEY = f"s3://{BUCKET_NAME}/{TRAIN_CHANNEL}"
S3_VALIDATION_OBJECT_KEY = f"s3://{BUCKET_NAME}/{VALIDATION_CHANNEL}"
S3_TEST_OBJECT_KEY = f"s3://{BUCKET_NAME}/{TEST_CHANNEL}"

# Define package label and record file names
PACKAGE_LABEL = "bird_200"
TRAIN_RECORD_FILE_NAME = f"{PACKAGE_LABEL}_{TRAIN_CHANNEL}.rec"
VALIDATION_RECORD_FILE_NAME = f"{PACKAGE_LABEL}_{VALIDATION_CHANNEL}.rec"
TEST_RECORD_FILE_NAME = f"{PACKAGE_LABEL}_{TEST_CHANNEL}.rec"

# Define input file path
BIRDS_200_REC_INPUT_FILE_PATH = INPUT_DIR / TEST_RECORD_FILE_NAME

# Define class IDs and number of training samples
CLASS_IDS = [17, 36, 47, 68, 73]
NUM_TRAINING_SAMPLES = 209

# Define resize size
RESIZE_SIZE = 256
