import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd
import base64

# Function to encode image to base64
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Paths to your images
main_bg_path = "mainimg1.jpg"  # Replace with your main background image path
sidebar_bg_path = "sb2.jpg"  # Replace with your sidebar background image path

# Encode images to base64
main_bg = get_base64_of_bin_file(main_bg_path)
sidebar_bg = get_base64_of_bin_file(sidebar_bg_path)

# CSS for separate backgrounds with adjustments for visibility
page_bg_img = f"""
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{main_bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* Sidebar background with transparency */
[data-testid="stSidebar"] {{
    background-image: url("data:image/jpeg;base64,{sidebar_bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-blend-mode: multiply;
    background-color: rgba(255, 255, 255, 0.8);  /* Add white transparency */
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);     /* Optional shadow for contrast */
}}
</style>
"""

# Inject CSS into Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Handwritten Digit Prediction App')
st.subheader("This app predicts the **HANDWRITTEN DIGITS**\nData is obtained from the [keras library](https://keras.io/api/datasets/mnist/)")


# Load the trained model
try:
    model = load_model('cnn_model.h5')
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error("Error loading model: Check the file path.")
    st.stop()

# Sidebar for uploading image
st.sidebar.header("Upload Image of Handwritten Digit *Only*")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

# Display waiting message until image is uploaded
if uploaded_file:
    st.success("Image uploaded successfully. Ready for prediction!")

    # Display the uploaded image
    img = Image.open(uploaded_file).convert('L').resize((28, 28))  # Grayscale and resize
    st.image(img, caption="Uploaded Image (Processed)", use_column_width=True)

    # Convert the image to a NumPy array for model prediction
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_flattened = img_array.reshape(1, 28, 28)  # Reshape for the model

    # Button to trigger prediction
    if st.button("Predict"):
        try:
            # Predict the digit
            prediction = model.predict(img_flattened)
            prediction_class = np.argmax(prediction)

            # Display the predicted digit
            label = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            st.subheader('Prediction')
            st.write(f'Predicted Digit: **{label[prediction_class]}**')

            st.subheader('Prediction Probabilities')
            st.write(prediction)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Waiting for image uploading...")
    
    st.image("smiley.gif", caption="Smiley GIF", width=300)
    # # Embed video with autoplay using HTML
    # video_path = "jass_dance.mp4"  # Replace with your video file path
    # video_html = f"""
    # <video width="100%" height="auto" controls autoplay>
    # <source src="{video_path}" type="video/mp4">
    # Your browser does not support the video tag.
    # </video>
    # """

    # st.markdown(video_html, unsafe_allow_html=True)
