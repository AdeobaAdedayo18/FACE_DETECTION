"""
Facial Emotion Recognition Model Training Script for Google Colab
Copy this entire script into a Colab notebook or Python cell.
"""

# ============================================================================
# STEP 1: INSTALL PACKAGES (Run this first in Colab)
# ============================================================================
# !pip install opencv-python pillow -q

# ============================================================================
# STEP 2: MOUNT GOOGLE DRIVE (if dataset is on Drive, uncomment below)
# ============================================================================
# from google.colab import drive
# drive.mount('/content/drive')
# dataset_path = '/content/drive/MyDrive/archive'  # Update to your path

# ============================================================================
# STEP 3: UPLOAD DATASET (if uploading directly, uncomment below)
# ============================================================================
# from google.colab import files
# import zipfile
# uploaded = files.upload()
# zip_filename = [f for f in uploaded.keys() if f.endswith('.zip')][0]
# with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
#     zip_ref.extractall('/content')
# dataset_path = '/content/archive'

# ============================================================================
# STEP 4: IMPORTS AND SETUP
# ============================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
from PIL import Image
import time

# Check GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print(f"GPU Device: {tf.config.list_physical_devices('GPU')[0]}")
    print("✅ GPU is available! Training will be faster.")

# Emotion labels
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

# ============================================================================
# STEP 5: DATASET PATH
# ============================================================================
# Set your dataset path here:
# - If uploaded via zip: dataset_path = '/content/archive'
# - If on Google Drive: dataset_path = '/content/drive/MyDrive/archive'
# - If you set it in Step 2 or 3, it's already defined above

# Uncomment and update if needed:
# dataset_path = '/content/archive'

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_images_from_folder(folder_path, emotion_label):
    """Load images from a specific emotion folder."""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return images, labels
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"  Loading {len(image_files)} images from {os.path.basename(folder_path)}...")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                pil_img = Image.open(img_path).convert('L')
                img = np.array(pil_img)
            
            img_resized = cv2.resize(img, (48, 48))
            img_normalized = img_resized.astype('float32') / 255.0
            
            images.append(img_normalized)
            labels.append(emotion_label)
            
        except Exception as e:
            continue
    
    return images, labels

def load_fer2013_data(data_path):
    """Load FER2013 dataset from folder structure."""
    print("=" * 60)
    print("LOADING FER2013 DATASET")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset folder not found at {data_path}")
        print("Please check the path or upload the dataset.")
        return None, None, None, None
    
    print(f"Found dataset at: {data_path}")
    
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"ERROR: 'train' or 'test' folders not found")
        return None, None, None, None
    
    # Load training images
    print("\n📂 Loading training images...")
    train_images = []
    train_labels = []
    
    start_time = time.time()
    for emotion_folder, label in EMOTION_FOLDER_MAP.items():
        emotion_path = os.path.join(train_path, emotion_folder)
        images, labels = load_images_from_folder(emotion_path, label)
        train_images.extend(images)
        train_labels.extend(labels)
    
    print(f"✓ Training images loaded in {time.time() - start_time:.2f} seconds")
    
    # Load test images
    print("\n📂 Loading test images...")
    test_images = []
    test_labels = []
    
    start_time = time.time()
    for emotion_folder, label in EMOTION_FOLDER_MAP.items():
        emotion_path = os.path.join(test_path, emotion_folder)
        images, labels = load_images_from_folder(emotion_path, label)
        test_images.extend(images)
        test_labels.extend(labels)
    
    print(f"✓ Test images loaded in {time.time() - start_time:.2f} seconds")
    
    # Convert to numpy arrays
    X_train = np.array(train_images, dtype='float32')
    y_train = np.array(train_labels, dtype='int32')
    X_test = np.array(test_images, dtype='float32')
    y_test = np.array(test_labels, dtype='int32')
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"\n{'='*60}")
    print(f"✓ Training samples: {len(X_train):,}")
    print(f"✓ Test samples: {len(X_test):,}")
    print(f"✓ Image shape: {X_train[0].shape}")
    print(f"{'='*60}\n")
    
    return X_train, y_train, X_test, y_test

# ============================================================================
# MODEL BUILDING FUNCTION
# ============================================================================
def build_model(input_shape=(48, 48, 1), num_classes=7):
    """Build CNN model for facial emotion recognition."""
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
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_model():
    """Main training function."""
    print("\n" + "=" * 60)
    print("FACIAL EMOTION RECOGNITION MODEL TRAINING")
    print("=" * 60 + "\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_fer2013_data(dataset_path)
    
    if X_train is None:
        print("\n❌ Failed to load dataset. Please check the path and try again.")
        return
    
    # Build model
    print("Building CNN model...")
    model = build_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'face_emotionModel.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print("This may take 20-60 minutes with GPU, or 2-4 hours with CPU only.")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time/60:.2f} minutes")
    
    # Load best weights if saved
    if os.path.exists('face_emotionModel.h5'):
        model.load_weights('face_emotionModel.h5')
        print("✓ Loaded best model weights")
    
    # Evaluate final model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Model saved as: face_emotionModel.h5")
    print("=" * 60)
    
    # Save final model
    model.save('face_emotionModel.h5')
    print("\n✅ Model saved successfully!")
    
    # Download instruction
    print("\n📥 To download the model in Colab, run:")
    print("   from google.colab import files")
    print("   files.download('face_emotionModel.h5')")

# ============================================================================
# RUN TRAINING
# ============================================================================
if __name__ == '__main__':
    # Make sure dataset_path is set
    if 'dataset_path' not in globals():
        print("⚠️  ERROR: dataset_path is not defined!")
        print("Please set dataset_path before running this script.")
        print("\nExample:")
        print("  dataset_path = '/content/archive'  # If uploaded via zip")
        print("  OR")
        print("  dataset_path = '/content/drive/MyDrive/archive'  # If on Drive")
    else:
        train_model()

