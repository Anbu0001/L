import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the Pre-trained Classification Model (Module 5 & 7)
model = load_model('models/antispoof_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Classification Labels
LABELS = ["Real Face", "Photo Attack", "Video Attack", "3D Mask"]

def detect_and_predict(frame):
    # Module 2: Detection and Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (160, 160)) # Resize for CNN
        face_roi = face_roi.astype("float") / 255.0 # Normalization
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Module 3 & 4 & 5: Feature Extraction & Classification
        preds = model.predict(face_roi)[0]
        j = np.argmax(preds)
        label = LABELS[j]
        
        # Module 6: Decision and Output
        color = (0, 255, 0) if label == "Real Face" else (0, 0, 255)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_and_predict(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)