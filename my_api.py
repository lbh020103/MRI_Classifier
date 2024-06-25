import os
import pickle
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# model = tf.keras.models.load_model('alzheimer_model.h5')
# model = keras.models.load_model('alzheimer_model.sav')
model = pickle.load(open('alzheimer_model.sav', 'rb'))

# Define label mapping
label_mapping = {
    0: 'Mild Demented',
    1: 'Moderate Demented',
    2: 'Non Demented',
    3: 'Very Mild Demented'
}

app.config['UPLOAD_FOLDER'] = "static"


@app.route('/', methods=['POST', 'GET'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
        try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Save image to file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(path_to_save)

                # Read image file
                img = Image.open(path_to_save)

                # Ensure the image has 3 color channels (RGB)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Preprocess the image to fit the model input requirements
                img = img.resize((256, 256))  # Example size, modify based on your model's requirements
                img_array = np.array(img) / 255.0  # Normalize if required

                # Ensure the input shape matches the expected model input shape
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Make prediction
                prediction = model.predict(img_array)

                # Extract the predicted index
                predicted_index = int(np.argmax(prediction, axis=1)[0])

                # Map the index to the corresponding label
                predicted_label = label_mapping[predicted_index]

                # Return the prediction and image file path to be displayed on the HTML page
                return jsonify({'prediction': f'Prediction: {predicted_label}', 'image_url': path_to_save, 'hide_upload_wrap': True})
        except Exception as e:
            return jsonify({'error': str(e), 'hide_upload_wrap': False}), 400
    else:
        return render_template('main.html')




# Start Backend
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='6868')
