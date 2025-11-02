from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sqlite3
import os

print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

app = Flask(__name__)
model = load_model('face_emotionModel.h5')

# Define emotion labels (adjust based on your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize simple database
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)')
    conn.close()

@app.route('/')
def home():
    return render_template('index.htm')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Read and preprocess the image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.expand_dims(img, axis=[0, -1])

    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]

    response_message = f"You look {emotion.lower()} today."
    if emotion.lower() == 'sad':
        response_message += " Why are you sad?"

    return response_message

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
