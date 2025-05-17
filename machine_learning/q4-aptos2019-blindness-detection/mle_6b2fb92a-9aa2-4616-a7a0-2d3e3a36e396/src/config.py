import os

# Paths
BASE_DIR = '/workspace/mle_6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396'
DATA_DIR = '/workspace/mle_dataset'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_images')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Model parameters
MODEL_NAME = 'efficientnet_b5'
NUM_CLASSES = 5
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.2
SEED = 42
NUM_FOLDS = 5

# Experiment ID
EXPERIMENT_ID = '6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396'
CONTROL_GROUP = 'control_group_partition_1'
RESULTS_FILE = f'results_{EXPERIMENT_ID}_{CONTROL_GROUP}.txt'