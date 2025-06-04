#!/usr/bin/env python
# Diabetic Retinopathy Detection using MobileNetV2
# APTOS 2019 Dataset - Simplified version

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score, f1_score
import cv2
import tensorflow as tf
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
else:
    logger.info("No GPU found. Using CPU.")

def parse_args():
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--train_images_dir', type=str, required=True, help='Path to train images directory')
    parser.add_argument('--test_images_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--val_samples', type=int, default=200, help='Number of validation samples to use')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum number of training samples to use')
    return parser.parse_args()

def preprocess_image(image_path, img_size):
    """
    Preprocess image for model input
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image {image_path}")
            return np.zeros((img_size, img_size, 3), dtype=np.float32)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop the black borders (simple circular crop)
        h, w = img.shape[:2]
        radius = min(h, w) // 2
        center_x, center_y = w // 2, h // 2
        
        # Create a circular mask
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        
        # Apply mask
        masked_img = img.copy()
        masked_img[~mask] = 0
        
        # Resize image
        img_resized = cv2.resize(masked_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        return img_normalized.astype(np.float32)
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return np.zeros((img_size, img_size, 3), dtype=np.float32)

def create_model(img_size):
    """
    Create and return a MobileNetV2 model
    """
    try:
        # Use MobileNetV2 as base model (lighter than ResNet50)
        base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(img_size, img_size, 3),
            alpha=0.75  # Use a smaller model (75% of filters)
        )
        
        # Add custom top layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(5, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        logger.info(f"Created MobileNetV2 model with input size {img_size}x{img_size}")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def train_model(model, train_generator, validation_generator, epochs, batch_size, output_dir):
    """
    Train the model
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        start_time = time.time()
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return model, history, training_time
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def evaluate_model(model, test_generator, test_df, output_dir):
    """
    Evaluate the model
    """
    try:
        # Predict on test data
        start_time = time.time()
        
        predictions = model.predict(
            test_generator, 
            verbose=1
        )
        
        inference_time = time.time() - start_time
        
        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        
        # If test_df has ground truth labels, calculate metrics
        if 'diagnosis' in test_df.columns:
            true_classes = test_df['diagnosis'].values
            
            # Calculate metrics
            accuracy = accuracy_score(true_classes, predicted_classes)
            kappa = cohen_kappa_score(true_classes, predicted_classes, weights='quadratic')
            f1 = f1_score(true_classes, predicted_classes, average='weighted')
            
            # Print metrics
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Quadratic Weighted Kappa: {kappa:.4f}")
            logger.info(f"F1 Score (weighted): {f1:.4f}")
            logger.info(f"Inference Time: {inference_time:.2f} seconds")
            
            # Generate classification report
            class_report = classification_report(true_classes, predicted_classes)
            logger.info("Classification Report:")
            logger.info(class_report)
            
            # Save confusion matrix
            try:
                cm = confusion_matrix(true_classes, predicted_classes)
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
                plt.close()
            except Exception as e:
                logger.error(f"Error creating confusion matrix: {e}")
            
            metrics = {
                'accuracy': accuracy,
                'kappa': kappa,
                'f1_score': f1,
                'training_time': None,
                'inference_time': inference_time
            }
        else:
            metrics = {
                'inference_time': inference_time
            }
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id_code': test_df['id_code'],
            'diagnosis': predicted_classes
        })
        
        # Save submission file
        submission_path = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
        
        return metrics, submission_df
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

class SimpleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, directory, x_col, y_col=None, batch_size=8, img_size=128, shuffle=True):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataframe))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Get the batch of dataframe rows
        batch_df = self.dataframe.iloc[batch_indexes]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_df), self.img_size, self.img_size, 3), dtype=np.float32)
        
        # Generate data
        for i, row_idx in enumerate(batch_indexes):
            try:
                row = self.dataframe.iloc[row_idx]
                img_path = os.path.join(self.directory, f"{row[self.x_col]}.png")
                img = preprocess_image(img_path, self.img_size)
                batch_x[i] = img
            except Exception as e:
                logger.error(f"Error processing image at index {row_idx}: {e}")
                batch_x[i] = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        
        if self.y_col is not None:
            batch_y = np.array(batch_df[self.y_col])
            return batch_x, batch_y
        else:
            return batch_x
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Load data
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        
        logger.info(f"Training data: {train_df.shape[0]} samples")
        logger.info(f"Test data: {test_df.shape[0]} samples")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Limit training data if specified
        if args.max_train_samples is not None and args.max_train_samples < train_df.shape[0]:
            logger.info(f"Limiting training data to {args.max_train_samples} samples")
            train_df = train_df.sample(n=args.max_train_samples, random_state=42)
        
        # Split training data into train and validation sets
        val_size = min(args.val_samples, int(train_df.shape[0] * 0.2))
        logger.info(f"Using {val_size} samples for validation")
        
        # Stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split
        train_subset, val_subset = train_test_split(
            train_df, 
            test_size=val_size,
            random_state=42,
            stratify=train_df['diagnosis'] if 'diagnosis' in train_df.columns else None
        )
        
        logger.info(f"Training subset: {train_subset.shape[0]} samples")
        logger.info(f"Validation subset: {val_subset.shape[0]} samples")
        
        # Create data generators
        train_generator = SimpleDataGenerator(
            train_subset,
            args.train_images_dir,
            'id_code',
            'diagnosis',
            batch_size=args.batch_size,
            img_size=args.img_size,
            shuffle=True
        )
        
        validation_generator = SimpleDataGenerator(
            val_subset,
            args.train_images_dir,
            'id_code',
            'diagnosis',
            batch_size=args.batch_size,
            img_size=args.img_size,
            shuffle=False
        )
        
        test_generator = SimpleDataGenerator(
            test_df,
            args.test_images_dir,
            'id_code',
            None if 'diagnosis' not in test_df.columns else 'diagnosis',
            batch_size=args.batch_size,
            img_size=args.img_size,
            shuffle=False
        )
        
        # Create model
        model = create_model(args.img_size)
        
        # Train model
        model, history, training_time = train_model(
            model,
            train_generator,
            validation_generator,
            args.epochs,
            args.batch_size,
            args.output_dir
        )
        
        # Evaluate model
        metrics, submission_df = evaluate_model(model, test_generator, test_df, args.output_dir)
        metrics['training_time'] = training_time
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
        
        # Plot training history
        if history is not None:
            try:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('Model Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'], loc='upper left')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, 'training_history.png'))
                plt.close()
            except Exception as e:
                logger.error(f"Error plotting training history: {e}")
        
        logger.info("Done!")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()