import os
import torch

# Paths
DATA_DIR = "/workspace/mle_dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
OUTPUT_DIR = "/workspace/mle_0083c3b8-243d-4eda-a884-57fddb81c9ce/output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "efficientnet_b4_model.pth")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_NAME = "efficientnet-b4"
NUM_CLASSES = 5  # 0, 1, 2, 3, 4 (diabetic retinopathy severity levels)

# Training configuration
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10
IMAGE_SIZE = 380  # EfficientNet-B4 recommended size

# Regularization configuration
DROPOUT_RATE = 0.2  # Standard dropout rate

# Augmentation configuration
AUGMENTATION_LEVEL = "standard"  # Options: "standard", "strong"