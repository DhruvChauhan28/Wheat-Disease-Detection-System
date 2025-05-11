import streamlit as st
import tensorflow as tf
import numpy as np
import pymysql  # for MySQL connection

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"D:\drive d\study material\3rd year study material\sem 6\design engineering\wheat_disease_final_model")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index

def get_cure_from_db(disease_name):
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',  # Change if different
            password='',  # Update if password is set
            database='wheat_disease_db'
        )
        cursor = connection.cursor()
        query = "SELECT cure FROM disease_cures WHERE disease_name = %s"
        cursor.execute(query, (disease_name,))
        result = cursor.fetchone()
        return result[0] if result else "Cure not found in database."
    except pymysql.MySQLError as e:
        return f"Database error: {e}"
    finally:
        if connection:
            connection.close()

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("Wheat DISEASE RECOGNITION SYSTEM")
    image_path = r"D:\drive d\study material\3rd year study material\sem 6\design engineering\design_project_code\home_page_image.jpg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Wheat Disease Recognition System! 🌿🔍

    Wheat Disease Cause:
    - **Fungi:** Fusarium, Septoria, Rusts
    - **Bacteria:** Bacterial streak, bunt
    - **Viruses:** Mosaic virus
    - **Environment:** Stress conditions

    ### Get Started
    Go to **Disease Recognition** and upload an image.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    Dataset consists of 14K images of healthy/diseased wheat leaves in 38 categories.
    - Train: 13105
    - Test: 700
    - Validation: 300
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if st.button("Show Image") and test_image:
        st.image(test_image, use_container_width=True)

    if st.button("Predict") and test_image:
        st.snow()
        st.write("Our Prediction")

        result_index = model_prediction(test_image)
        class_name = [
            'Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common root rot',
            'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite',
            'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust'
        ]

        predicted_disease = class_name[result_index]
        st.success(f"Model is predicting it's a **{predicted_disease}**")
        cure = get_cure_from_db(predicted_disease)
        st.info(f"**Suggested Cure:** {cure}")
