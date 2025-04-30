from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K
import cv2
from auth.routes import auth

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========== MODEL LOADING ==========
try:
    K.clear_session()  # Clear cache once during app startup
    model = load_model(os.path.join('models', 'blood_group_cnn_v2.h5'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Correct Class Order Mapping
model_class_indices = {
    0: 'A+',
    1: 'A-',
    2: 'AB+',
    3: 'AB-',
    4: 'B+',
    5: 'B-',
    6: 'O+',
    7: 'O-'
}

# ========== UTILITY FUNCTION ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (150, 150))

    # Improved clarity with normalized RGB image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure correct channel order
    img_array = img / 255.0

    return np.expand_dims(img_array, axis=0)

# Register Blueprints
app.register_blueprint(auth, url_prefix='/auth')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('fingerprint')

        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                img_array = preprocess_image(filepath)

                if model:
                    prediction = model.predict(img_array)
                    predicted_class = np.argmax(prediction)
                    confidence = round(np.max(prediction) * 100, 2)

                    result = model_class_indices.get(predicted_class, "Unknown")

                    # Cleanup uploaded files after prediction
                    os.remove(filepath)

                    return render_template('result.html', result=result, confidence=confidence)
                else:
                    flash('Model is not loaded properly. Check console for details.', 'danger')
                    return redirect(url_for('predict'))

            except Exception as e:
                flash(f'Error processing image: {e}', 'danger')
                return redirect(url_for('predict'))

        flash('No file selected or invalid file format. Please upload a valid image.', 'warning')
        return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
