import os

# Define paths
CURRENT_PATH = os.path.dirname(__file__)
IMG_PATH = os.path.join(CURRENT_PATH, r'../data/processed/train-tif-v2')
LABEL_PATH = os.path.join(CURRENT_PATH, r'../data/processed/train_v2.csv')
TRAIN_DATA_PATH = os.path.join(CURRENT_PATH, r'../data/processed/train_set.txt')
VALIDATION_DATA_PATH = os.path.join(CURRENT_PATH, r'../data/processed/validation_set.txt')

# Labels lookup table
LABELS = {'primary': 0,
          'clear': 1,
          'agriculture': 2,
          'road': 3,
          'water': 4,
          'partly_cloudy': 5,
          'cultivation': 6,
          'habitation': 7,
          'haze': 8,
          'cloudy': 9,
          'bare_ground': 10,
          'selective_logging': 11,
          'artisinal_mine': 12,
          'blooming': 13,
          'slash_burn': 14,
          'blow_down': 15,
          'conventional_mine': 16}

LABELS_SIZE = 17

# Train-validation separation parameters
TRAIN_VALIDATION_RATIO = 0.8
SEED = 448

# Raw image shape
RAW_IMAGE_WIDTH = 256
RAW_IMAGE_HEIGHT = 256
RAW_IMAGE_DEPTH = 4

# Training parameters
DATA_AUGMENTATION = False
NORMALIZE_IMAGE = True

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 4

BATCH_SIZE = 128
REGULARIZER = 0.0

INITIAL_LEARNING_RATE = 0.01
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 30000
NUM_EPOCHS_PER_DECAY = 10
LEARNING_RATE_DECAY_FACTOR = 0.1

MOVING_AVERAGE_DECAY = 0.9999

# Log options
LOG_PATH = os.path.join(CURRENT_PATH, r'../reports/planet_train')
MAX_STEPS = 5000  # number of batches to run
LOG_DEVICE_PLACEMENT = False
LOG_FREQUENCY = 10

# Validation parameters
EVALUATION_PATH = os.path.join(CURRENT_PATH, r'../reports/planet_validation')

NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION = 5000
EVALUATION_NUM_EXAMPLES = 5000
RUN_ONCE = False
EVALUATION_INTERVAL_SECS = 60 * 5
