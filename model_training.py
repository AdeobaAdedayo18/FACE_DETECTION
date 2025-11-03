"""
Facial Emotion Recognition Model Training Script
This script trains a CNN model on the FER2013 dataset to detect emotions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
from PIL import Image

# Emotion labels corresponding to FER2013 dataset
# Note: Folder names are lowercase, we'll map them correctly
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_FOLDER_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

def load_images_from_folder(folder_path, emotion_label):
    """
    Load all images from a specific emotion folder.
    
    Args:
        folder_path: Path to the emotion folder (e.g., 'train/angry')
        emotion_label: Integer label for the emotion (0-6)
    
    Returns:
        List of preprocessed images (48x48 grayscale, normalized)
        List of labels (all same value: emotion_label)
    """
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return images, labels
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"  Loading {len(image_files)} images from {os.path.basename(folder_path)}...")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                # Try with PIL if OpenCV fails
                pil_img = Image.open(img_path).convert('L')
                img = np.array(pil_img)
            
            # Resize to 48x48 (FER2013 input size)
            img_resized = cv2.resize(img, (48, 48))
            
            # Normalize pixel values to 0-1 range
            img_normalized = img_resized.astype('float32') / 255.0
            
            images.append(img_normalized)
            labels.append(emotion_label)
            
        except Exception as e:
            print(f"    Warning: Could not load {img_file}: {str(e)}")
            continue
    
    return images, labels

def load_fer2013_data(data_path='../Downloads/archive'):
    """
    Load and preprocess FER2013 dataset from folder structure.
    
    Expected folder structure:
    archive/
        train/
            angry/
            disgust/
            fear/
            happy/
            neutral/
            sad/
            surprise/
        test/
            angry/
            disgust/
            fear/
            happy/
            neutral/
            sad/
            surprise/
    """
    print("Loading FER2013 dataset from folder structure...")
    
    # Try different possible paths
    possible_paths = [
        data_path,
        '../Downloads/archive',
        'C:/Users/HP/Downloads/archive',
        os.path.join(os.path.dirname(__file__), '../Downloads/archive')
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print(f"ERROR: Dataset folder not found!")
        print("Please ensure the dataset is in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nThe dataset should have 'train' and 'test' folders with emotion subfolders.")
        return None, None, None, None
    
    print(f"Found dataset at: {dataset_path}")
    
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"ERROR: 'train' or 'test' folders not found in {dataset_path}")
        return None, None, None, None
    
    # Load training images
    print("\nLoading training images...")
    train_images = []
    train_labels = []
    
    for emotion_folder, label in EMOTION_FOLDER_MAP.items():
        emotion_path = os.path.join(train_path, emotion_folder)
        images, labels = load_images_from_folder(emotion_path, label)
        train_images.extend(images)
        train_labels.extend(labels)
    
    # Load test images
    print("\nLoading test images...")
    test_images = []
    test_labels = []
    
    for emotion_folder, label in EMOTION_FOLDER_MAP.items():
        emotion_path = os.path.join(test_path, emotion_folder)
        images, labels = load_images_from_folder(emotion_path, label)
        test_images.extend(images)
        test_labels.extend(labels)
    
    # Convert to numpy arrays
    X_train = np.array(train_images, dtype='float32')
    y_train = np.array(train_labels, dtype='int32')
    X_test = np.array(test_images, dtype='float32')
    y_test = np.array(test_labels, dtype='int32')
    
    # Add channel dimension: (N, 48, 48) -> (N, 48, 48, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"\n✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    print(f"✓ Image shape: {X_train[0].shape}")
    
    return X_train, y_train, X_test, y_test

def build_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Build a CNN model for facial emotion recognition.
    
    Architecture:
    - Multiple convolutional layers with increasing filters
    - MaxPooling layers for downsampling
    - Dropout layers to prevent overfitting
    - Dense layers for final classification
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer (7 emotions)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """
    Main training function that loads data, builds model, trains it, and saves it.
    """
    print("=" * 50)
    print("FACIAL EMOTION RECOGNITION MODEL TRAINING")
    print("=" * 50)
    
    # Load data from folder structure
    # Update this path if your dataset is in a different location
    X_train, y_train, X_test, y_test = load_fer2013_data('C:/Users/HP/Downloads/archive')
    
    if X_train is None:
        return
    
    # Build model
    print("\nBuilding CNN model...")
    model = build_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        keras.callbacks.ModelCheckpoint(
            'face_emotionModel.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    print("This may take a while (30+ minutes depending on your hardware)...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model weights if saved
    if os.path.exists('face_emotionModel.h5'):
        model.load_weights('face_emotionModel.h5')
    
    # Evaluate final model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Model saved as 'face_emotionModel.h5'")
    print("\nTraining completed!")

if __name__ == '__main__':
    train_model()

