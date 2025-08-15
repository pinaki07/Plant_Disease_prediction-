import base64
import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px

# --- Apply Background ---

df = px.data.iris()

# Function to convert local image to base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load background images
main_bg = get_img_as_base64("background.jpg")         # Main background image
sidebar_bg = get_img_as_base64("side_bar.jpg")        # Sidebar background image

# CSS styling with overlay + text color enhancements
page_bg_img = f"""
<style>
/* Main background with dark overlay */
[data-testid="stAppViewContainer"] > .main {{
background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                  url("data:image/jpeg;base64,{main_bg}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
}}

/* Sidebar background with light overlay */
[data-testid="stSidebar"] > div:first-child {{
background-image: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                  url("data:image/jpeg;base64,{sidebar_bg}");
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

/* Transparent header */
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

/* Toolbar positioning */
[data-testid="stToolbar"] {{
right: 2rem;
}}

/* MAIN AREA: make all headings and text white */
.main .block-container, .main h1, .main h2, .main h3, .main h4, .main h5, .main h6, .main p, .main li, .main div {{
    color: white !important;
}}

/* SIDEBAR: make text black */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] h5, 
[data-testid="stSidebar"] h6, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] div {{
    color: black !important;
}}
/* Make buttons green on Disease Recognition page */
div.stButton > button:first-child {{
    background-color: #28a745 !important;  /* Bootstrap green */
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.5em;
    font-weight: bold;
    transition: background-color 0.3s ease;
}}
div.stButton > button:first-child:hover {{
    background-color: #218838 !important;
}}

</style>
"""

# Inject custom CSS into Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- TensorFlow Model Prediction Function ---

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')

def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return result_index, confidence

# --- Sidebar ---
st.sidebar.markdown("## 🌿 Plant Health AI")
st.sidebar.markdown("### 🔍 Navigation")
st.sidebar.markdown("#### 🔽 Choose a Page")
app_mode = st.sidebar.selectbox(
    "",
    ["🏠 Home", "📖 About", "🧪 Disease Recognition"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 Developed by IEM Student")

# --- Home Page ---
if app_mode == "🏠 Home":
    st.header("🌿 Plant Disease Recognition System")
    st.markdown("""
    ## 👋 Welcome!

    This is your one-stop solution to identify plant diseases using cutting-edge Machine Learning technology. Upload a leaf image, and our system will analyze it to detect any disease—quickly and accurately.

    ---

    ## 🚀 How It Works
    1. **Go to the Disease Recognition page.**
    2. **Upload an image** of a plant leaf showing symptoms.
    3. **Our AI model** analyzes the image to predict the disease.
    4. **Get instant results** and take action to protect your crops!

    ---

    ## ✅ Why Use This System?
    - 🌱 **Trained with Real Data:** Model trained on thousands of real plant disease images.
    - ⚡ **Fast & Accurate:** Get results in seconds with high prediction accuracy.
    - 🧠 **ML-Powered:** Built using a powerful Convolutional Neural Network (CNN).
    - 🖥️ **Simple Interface:** No need for technical skills — just upload and see results!

    ---

    ## 🧪 Ready to Detect a Disease?
    👉 Navigate to **Disease Recognition** from the sidebar to upload your image.

    ---

    ## 👥 About This Project
    Learn more about us in the **About** section.

    ---
    """)

# --- About Page ---
elif(app_mode == "📖 About"):
    st.header("📖 About")
    st.markdown("""
    #### 📂 About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.  
    This dataset consists of about **87K RGB images** of healthy and diseased crop leaves which are categorized into **38 different classes**.  
    The total dataset is divided into an **80/20 ratio** for training and validation while preserving the directory structure.  
    A new directory containing **33 test images** is created later for prediction purposes.

    #### 📊 Content
    1. **train** (70,295 images)  
    2. **test** (33 images)  
    3. **validation** (17,572 images)
    """)

    st.markdown("""
    ---

    #### 🎯 Project Objective
    This project is developed by a B.Tech 4th-year student from IEM Kolkata as a final year academic project. The goal is to assist farmers and researchers 
    in identifying plant diseases using deep learning. The aim of this project is to **develop a web-based tool** that can accurately detect plant diseases
    from images using a trained machine learning model. This system helps farmers and agriculturists take **faster and more informed decisions** for crop treatment.

    #### 🛠️ Tools & Technologies Used
    - **Python 3.10**
    - **TensorFlow/Keras** – for model development and prediction
    - **Streamlit** – to build an interactive web interface
    - **OpenCV & Matplotlib** – for image processing and visualization
    - **NumPy & Pandas** – for data handling

    #### 🧠 Model Architecture
    The model is based on a **Convolutional Neural Network (CNN)** trained on the dataset above.  
    The final trained model achieves:
    - ✅ **~90% validation accuracy**
    - ⚠️ May vary slightly depending on environment and hardware

    #### 📌 How to Use
    - Go to the **Disease Recognition** tab from the sidebar
    - Upload a plant image (leaf) to detect its disease class
    - View predictions instantly

    ---
    """)


# --- Disease Recognition Page ---
elif app_mode == "🧪 Disease Recognition":
    st.header("🌿 Disease Recognition")
    st.markdown("Upload a plant leaf image and let the model detect the disease.")

    test_image = st.file_uploader("📷 Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True, caption="Uploaded Image")

        if st.button("Predict"):
            with st.spinner("Analyzing Image... Please wait."):
                st.balloons()
                try:
                    result_index, confidence = model_prediction(test_image)
                    class_names = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]

                    prediction_label = class_names[result_index]
                    st.success(f"🩺 Model Prediction: **{prediction_label}** ({confidence*100:.2f}% confidence)")

                except Exception as e:
                    st.error("⚠️ Error in prediction. Make sure the uploaded file is a valid image.")
                    st.exception(e)
    else:
        st.info("👈 Please upload a plant leaf image to get started.")
