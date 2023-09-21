import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import math

# Load the pre-trained model
model = tf.keras.models.load_model("./models/1")

# Define your actual class names here (e.g., ["class1", "class2", ...])
class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


# Function to predict class and confidence for an image
def predict_single_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence


# Streamlit app
st.title("Potato Disease Classification")

# Sidebar options
option = st.sidebar.selectbox(
    "Choose an option", ["Predict a single image", "Batch Upload"]
)

if option == "Predict a single image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image:", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image with specified width and height
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        predicted_class, confidence = predict_single_image(uploaded_file)
        st.write(
            f"{predicted_class.replace('Potato___', '')}, Confidence: {confidence}%"
        )

if option == "Batch Upload":
    st.sidebar.info(
        "Please upload multiple images. Click 'Process' after uploading all images."
    )
    uploaded_files = st.file_uploader(
        "Upload multiple images:",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )
    process_button = st.button("Process")

    if uploaded_files and process_button:
        if len(uploaded_files) == 0:
            st.write("No files uploaded. Please upload images before processing.")
        else:
            batch_results = []
            num_columns = 3  # Number of columns for displaying images in a matrix

            for file in uploaded_files:
                predicted_class, confidence = predict_single_image(file)
                batch_results.append(
                    {
                        "File Name": file.name,
                        "Predicted Class": predicted_class.replace("Potato___", ""),
                        "Confidence (%)": confidence,
                    }
                )

            # Create a DataFrame from batch results
            batch_df = pd.DataFrame(batch_results)

            # Display the results in a table
            st.write("Batch Prediction Results:")
            st.dataframe(
                batch_df.style.highlight_max(axis=0, subset=["Confidence (%)"])
            )

            # Generate a CSV file for batch results
            csv_file = batch_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv_file,
                file_name="batch_results.csv",
                key="batch_csv",
            )

            # Display uploaded images in a matrix
            st.write("Uploaded Images:")
            num_images = len(uploaded_files)
            num_rows = math.ceil(num_images / num_columns)

            for i in range(num_rows):
                st.write("<div style='display: flex;'>", unsafe_allow_html=True)
                for j in range(num_columns):
                    index = i * num_columns + j
                    if index < num_images:
                        file = uploaded_files[index]
                        predicted_class = batch_results[index]["Predicted Class"]
                        confidence = batch_results[index]["Confidence (%)"]
                        img = load_img(file, target_size=(64, 64))
                        st.image(
                            img,
                            caption=f"Predicted: {predicted_class}\nConfidence: {confidence}%",
                            use_column_width=False,
                            width=128,
                        )
                    else:
                        st.write(
                            "<div style='width: 128px;'></div>", unsafe_allow_html=True
                        )
                st.write("</div>", unsafe_allow_html=True)
