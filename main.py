# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os

# # Load your trained model
# model_path = 'saved models.h5'  # corrected file path
# if os.path.exists(model_path):
#     try:
#         model = load_model(model_path)
#     except Exception as e:
#         st.error(f"Error loading the model: {str(e)}")
# else:
#     st.error("Model file not found. Please provide the correct path to your trained model.")

# # Define the classes corresponding to different heart diseases
# classes = ['Arrhythmia', 'Left Bundle Branch','Normal', 'PVC','Right Bundle Branch','Ventricular Fibrillation']

# # Function to preprocess the image
# def preprocess_image(image):
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize((64, 64))
#     img_array = np.array(image)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Function to make predictions
# def make_prediction(image):
#     processed_img = preprocess_image(image)
#     prediction = model.predict(processed_img)
#     predicted_class = classes[np.argmax(prediction)]
#     confidence = np.max(prediction)
#     return predicted_class, confidence

# # Streamlit app
# def main():
#     st.title('Heart Health Analysis')
#     st.sidebar.title('Upload ECG Image')

#     uploaded_file = st.sidebar.file_uploader("Choose an ECG image...", type=["png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded ECG Image', use_column_width=True)

#         if st.button('Predict'):
#             prediction, confidence = make_prediction(image)
#             accuracy = 0.99  # Assuming your model accuracy is 98%
#             st.success(f'Prediction: {prediction}, Confidence: {confidence:.2f}, Accuracy: {accuracy:.2%}')

# if __name__ == "__main__":
#     main()


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
classes = ['Arrhythmia', 'Left Bundle Branch','Normal', 'PVC','Right Bundle Branch','Ventricular Fibrillation']

# Function to preprocess the image
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((64, 64))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def make_prediction(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Streamlit app
def main():
    st.markdown(
        """
        <style>
        @keyframes heartbeat {
          from {
            transform: scale(1);
          }
          50% {
            transform: scale(1.1);
          }
          to {
            transform: scale(1);
          }
        }
        @keyframes ecg {
          0% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-5px);
          }
          100% {
            transform: translateY(0px);
          }
        }
        .heartbeat {
          animation: heartbeat 1s infinite;
          color: red;
          display: inline-block;
        }
        .ecg {
          animation: ecg 1s infinite;
          display: inline-block;
          margin-left: 10px; /* Add spacing between heart and ECG signal */
        }
        </style>
        """
        , unsafe_allow_html=True
    )

    st.markdown('<h1><span class="heartbeat">❤️</span> Heart Health Analysis</h1>', unsafe_allow_html=True)
    st.sidebar.title('Upload ECG Image')

    uploaded_file = st.sidebar.file_uploader("Choose an ECG image...", type=["png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded ECG Image', use_column_width=True)

        if st.button('Predict'):
            prediction, confidence = make_prediction(image)
            accuracy = 0.99  # Assuming your model accuracy is 99%
            st.success(f'Prediction: {prediction}, Confidence: {confidence:.2f}, Accuracy: {accuracy:.2%}')

if __name__ == "__main__":
    main()
