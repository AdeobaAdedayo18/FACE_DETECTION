"""
Flask Web Application for Facial Emotion Detection
This app allows users to upload images and detect emotions from faces.
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
import os
from datetime import datetime
import io
import base64
 

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Emotion labels (matching FER2013 dataset)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion messages mapping
EMOTION_MESSAGES = {
    'Angry': "You look angry. What's making you upset?",
    'Disgust': "You appear disgusted. What's bothering you?",
    'Fear': "You look fearful. Is everything okay?",
    'Happy': "You're smiling! That's wonderful to see!",
    'Sad': "You are frowning. Why are you sad?",
    'Surprise': "You look surprised! What caught you off guard?",
    'Neutral': "You have a neutral expression. How are you feeling?"
}

# Global variable to store the loaded model
model = None

 

def load_model():
    """Load the trained emotion recognition model."""
    global model
    if model is None:
        model_path = 'face_emotionModel.h5'
        if os.path.exists(model_path):
            print("Loading emotion recognition model...")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model file {model_path} not found!")
            print("Please train the model first using model_training.py")
    return model

def init_database():
    """Initialize SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            student_id TEXT,
            submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create images table (stores image data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_data BLOB,
            emotion_detected TEXT,
            confidence REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def preprocess_image(image_file):
    """
    Preprocess uploaded image for emotion prediction.
    Converts image to grayscale 48x48 format expected by the model.
    """
    # Read image file
    image_bytes = image_file.read()
    image_file.seek(0)  # Reset file pointer
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        # Try using PIL if OpenCV fails
        image_file.seek(0)
        pil_image = Image.open(image_file)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade (fallback if face not detected, use entire image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Use the largest detected face
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = gray[y:y+h, x:x+w]
    else:
        # If no face detected, use center crop of image
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w)
        start_x = max(0, center_x - crop_size // 2)
        start_y = max(0, center_y - crop_size // 2)
        face_roi = gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize to 48x48 (FER2013 input size)
    face_resized = cv2.resize(face_roi, (48, 48))
    
    # Normalize pixel values to 0-1 range
    face_normalized = face_resized.astype('float32') / 255.0
    
    # Add batch and channel dimensions: (48, 48) -> (1, 48, 48, 1)
    face_final = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=-1)
    
    return face_final, image_bytes

def predict_emotion(image_file):
    """Predict emotion from uploaded image."""
    model = load_model()
    if model is None:
        return None, None
    
    try:
        # Preprocess image
        processed_image, image_bytes = preprocess_image(image_file)
        
        # Predict emotion
        predictions = model.predict(processed_image, verbose=0)
        emotion_index = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_index])
        emotion = EMOTIONS[emotion_index]
        
        return emotion, confidence
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def save_to_database(name, email, student_id, image_bytes, emotion, confidence):
    """Save user information and image to database."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    try:
        # Insert user information
        cursor.execute('''
            INSERT INTO users (name, email, student_id)
            VALUES (?, ?, ?)
        ''', (name, email, student_id))
        
        user_id = cursor.lastrowid
        
        # Insert image and emotion data
        cursor.execute('''
            INSERT INTO images (user_id, image_data, emotion_detected, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_id, sqlite3.Binary(image_bytes), emotion, confidence))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Database error: {str(e)}")
        return False
    finally:
        conn.close()

@app.route('/')
def index():
    """Render the main page with the form."""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission and emotion detection."""
    try:
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        student_id = request.form.get('student_id', '').strip()
        image_file = request.files.get('image')
        
        # Validate inputs
        if not name or not email:
            flash('Please fill in all required fields (Name and Email).')
            return redirect(url_for('index'))
        
        if not image_file or image_file.filename == '':
            flash('Please upload an image.')
            return redirect(url_for('index'))
        
        # Check if file is an image
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in image_file.filename or \
           image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            flash('Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP).')
            return redirect(url_for('index'))
        
        # Read image bytes for database storage
        image_bytes = image_file.read()
        image_file.seek(0)  # Reset for prediction
        
        # Predict emotion
        emotion, confidence = predict_emotion(image_file)
        
        if emotion is None:
            flash('Error detecting emotion. Please try another image or ensure the model is trained.')
            return redirect(url_for('index'))
        
        # Save to database
        success = save_to_database(name, email, student_id, image_bytes, emotion, confidence)
        
        if not success:
            flash('Error saving data to database. Emotion detected but not saved.')
        
        # Get emotion message
        emotion_message = EMOTION_MESSAGES.get(emotion, f"You appear {emotion.lower()}.")
        
        # Render result page
        return render_template('index.html', 
                             result=True,
                             name=name,
                             emotion=emotion,
                             emotion_message=emotion_message,
                             confidence=f"{confidence * 100:.2f}%")
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Initialize database on startup
    init_database()
    
    # Load model on startup
    load_model()
    
    # Get port from environment variable (Railway provides this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run Flask app
    print("\n" + "=" * 50)
    print("FACIAL EMOTION DETECTION WEB APP")
    print("=" * 50)
    print("Starting Flask server...")
    print(f"Server running on port {port}")
    print("=" * 50 + "\n")
    
    # Disable debug mode in production (Railway sets FLASK_ENV)
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

