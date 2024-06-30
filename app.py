from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('xception_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        img = Image.open(file)
        img = img.resize((224, 224))  # Corrected to pass a tuple
        img_array = image.img_to_array(img) / 255.0  # Normalize if required
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)

        # Post-process and interpret results
        predicted_class = np.argmax(predictions, axis=-1)
        
        print("Predicted class:", predicted_class)
        if predicted_class == 0:
            predicted_class = 'Cataract'  # Corrected spelling
        elif predicted_class == 1:  # Corrected syntax error
            predicted_class = 'Diabetic Retinopathy'
        elif predicted_class == 2:  # Corrected syntax error
            predicted_class = 'Glaucoma'
        else:
            predicted_class = 'Normal Eye'
        # Render a template and pass the predicted_class to it
        return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)