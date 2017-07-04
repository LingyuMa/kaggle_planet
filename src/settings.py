import os

# Define paths
CURRENT_PATH = os.path.dirname(__file__)
IMG_PATH = os.path.join(CURRENT_PATH, r'../data/processed/train-tif-v2')
LABEL_PATH = os.path.join(CURRENT_PATH, r'../data/processed/train_v2.csv')
TRAIN_DATA_PATH = os.path.join(CURRENT_PATH, r'../data/processed/train_set.txt')
VALIDATION_DATA_PATH = os.path.join(CURRENT_PATH, r'../data/processed/validation_set.txt')

# Labels lookup table
WEATHER_LABELS = {'clear': 0,
                  'partly_cloudy': 1,
                  'haze': 2,
                  'cloudy': 3}

COMMON_LABELS = {'primary': 0,
                 'agriculture': 1,
                 'road': 2,
                 'water': 3,
                 'habitation': 4,
                 'cultivation': 5,
                 'bare_ground': 6}

RARE_LABELS = {'selective_logging': 0,
               'artisinal_mine': 1,
               'blooming': 2,
               'slash_burn': 3,
               'blow_down': 4,
               'conventional_mine': 5}

AUGMENT_FLAG = {'partly_cloudy': 'flip_1',
                'haze': 'rotate_45',
                'cloudy': 'rotate_45'}  # to be added

LABELS = [WEATHER_LABELS, COMMON_LABELS, RARE_LABELS]

# Add penalty to false negative
LOSS_WEIGHT = None

# Specify which network to train
NETWORK_ID = 0

# Train-validation separation parameters
TRAIN_VALIDATION_RATIO = 0.8
SEED = 448

# Training parameters
DATA_AUGMENTATION = False  # flip only
NORMALIZE_IMAGE = False
ZERO_MEAN_IMAGE = True

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 4

RAW_IMAGE_WIDTH = 256
RAW_IMAGE_HEIGHT = 256
RAW_IMAGE_DEPTH = 4

WEIGHT_DECAY_FC = 0.0002
WEIGHT_DECAY = 0.0002

BATCH_SIZE = 128

INITIAL_LEARNING_RATE = 0.01
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 70750
NUM_EPOCHS_PER_DECAY = 10
LEARNING_RATE_DECAY_FACTOR = 0.1

MOVING_AVERAGE_DECAY = 0.9999

# Log options
LOG_PATH = os.path.join(CURRENT_PATH, r'../reports/planet_train')
MAX_STEPS = 21000  # number of batches to run
LOG_DEVICE_PLACEMENT = False
LOG_FREQUENCY = 10

# Validation parameters
EVALUATION_PATH = os.path.join(CURRENT_PATH, r'../reports/planet_validation')

NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION = 33000
EVALUATION_NUM_EXAMPLES = 33000
RUN_ONCE = False
EVALUATION_INTERVAL_SECS = 60 * 5
