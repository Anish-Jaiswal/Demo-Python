import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load your trained model
model_path = 'saved models.h5'  # corrected file path
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
else:
    st.error("Model file not found. Please provide the correct path to your trained model.")

# Define the classes corresponding to different heart diseases
#'Premature Ventricular Contraction', 'Normal', 'Left Bundle Branch', 'Right Bundle Branch','Arrhythmia','Ventricular Fibrillation'
classes = ['Normal', 'Left Bundle Branch','Right Bundle Branch', 'PVC','Arrhythmia','Ventricular Fibrillation']

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB if it's not already in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image to match model input shape
    image = image.resize((64, 64))
    # Convert image to numpy array
    img_array = np.array(image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to make predictions
def make_prediction(image):
    # Preprocess the image
    processed_img = preprocess_image(image)
    # Make prediction using the loaded model
    prediction = model.predict(processed_img)
    # Get the predicted class label
    predicted_class = classes[np.argmax(prediction)]
    # Get the confidence score
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit app
def main():
    st.title('Heart Health Analysis')
    st.sidebar.title('Upload ECG Image')

    uploaded_file = st.sidebar.file_uploader("Choose an ECG image...", type=["png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded ECG Image', use_column_width=True)

        # Make prediction when button is clicked
        if st.button('Predict'):
            # Make prediction
            prediction, confidence = make_prediction(image)
            st.success(f'Prediction: {prediction}, Confidence: {confidence:.2f}')

if __name__ == "__main__":
    main()
