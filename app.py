import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image
import io
import time


st.markdown(
        """
        <style>
        .stApp {
                background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxxkV_AVtgPh8j2gfH_UpwKPodmiQaXiarTw&s");
                background-size: cover;
                background-position: center;
        }
        .stMainBlockContainer.ea3mdgi5{
                max-width:100%;
                padding: 0;
                margin: 0;
        }
        header {
                color: ; black
                padding: 10px;
        }
        .header-background{
                background-image: url("https://www.shutterstock.com/image-photo/xray-image-ankle-fracture-blue-260nw-2253312433.jpg");
                background-size: cover;
                background-position: center;
                height: 100vh;
                text-align: end;
                align-content: center;
                padding-right: 20px;
                text-decoration:none;
                clip-path: inset(0 0 35px 0);
        }
        .hero-section > p{
                padding-right: 20px;
        }
        .stFileUploader {
                width: 70%;
                margin: 0 auto;
        }
        .section-title {
                color:white;
                font-size: 30px;
                font-weight: bold;
                margin-top: 40px;
                width: 70%;
                margin: auto;
        }
        .section-subtitle {
                color:white;
                font-size: 18px;
                font-weight: normal;
                margin-top: -10px;
                width: 70%;
                margin: auto;
                margin-bottom:10px;
        }
        .decoration{
                text-align: center;
                text-decoration: overline;
                margin-top: 3%;
                font-size: 30px;
                font-weight: 800;
        }
        .stHorizontalBlock.st-emotion-cache-ocqkz7.e1f1d6gn5{
                width: 70%;
                margin: auto;
        }
        .st-emotion-cache-1xf0csu.e115fcil1 > img{
                height: 300px;
        }
        #root > div:nth-child(1) > div.withScreencast > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 > div > div > div > div:nth-child(5) > div.stColumn.st-emotion-cache-fplge5.e1f1d6gn3 > div{
                width: 100%;
        }
        .footer {
                color: #ffff;
                background-color:black;
                text-align: center;
                margin-top: 50px;
                font-size: 16px;
                padding:12px;
        }
        </style>
        """,
        unsafe_allow_html=True
)


# CSS for floating effect
st.markdown(
    """
    <style>
    /* Floating animation */
    @keyframes floatIn {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    /* Apply animation to wrapper div */
    .floating-element {
        animation: floatIn 1.5s ease-out forwards;
        opacity: 0;
    }

    /* Delay for staggered animation */
    .floating-element-1 { animation-delay: 0.2s; }
    .floating-element-2 { animation-delay: 0.4s; }
    .floating-element-3 { animation-delay: 1.4s; }
    .floating-element-4 { animation-delay: 1.8s; }

    </style>
    """,
    unsafe_allow_html=True
)


# header
st.markdown('''<header class='header-background full-width-header '>
                        <div class='hero-section floating-element floating-element-1'><h1>Fracture Detection in X-ray Images</h1>
                                <p class='hero-section floating-element floating-element-2'>An AI-powered application to assist in detecting fractures from X-ray images.</p>
                        </div>
                </header>''',
               unsafe_allow_html=True)



st.markdown("<div class='decoration floating-element floating-element-3'>Upload an X-ray Image </div>",unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

model = load_model("/Users/harshsoni2003/Downloads/Final/model/recognition_model.keras")

data_label= ['fractured', 'not fractured']

image_height = 256
image_width = 256

# Run classification if an image is uploaded
if uploaded_file is not None:
        try:
                # Columns layout
                col1, col2 = st.columns([3, 2])

                with col1:
                        # spinner
                        with st.spinner("Processing the uploaded image..."):
                                time.sleep(1)

                        # reading image
                        image = Image.open(io.BytesIO(uploaded_file.read()))
                        if image.mode != "RGB":
                                image = image.convert("RGB")

                        img_arr = np.array(image.resize((image_height,image_width)))
                        img_arr = np.array(img_arr.reshape((1,image_height,image_width,3)))

                        predict = model.predict(img_arr)

                        # Use softmax to get probabilities for each class
                        probabilities = tf.nn.softmax(predict)

                        # Get the predicted class
                        predicted_class = np.argmax(probabilities)

                        # Display the uploaded image
                        st.image(image, caption="Uploaded X-ray Image")

                with col2:
                        # Display Result
                        st.subheader("Detection Result")

                        st.write("The bone is : ", data_label[predicted_class] )

                        st.write("This result is generated by our deep learning model, trained on X-ray images.")
        except Exception as e:
                st.error(f"An error occured :{e}")




# Example X-ray Images Section
st.markdown("<div class='section-title floating-element floating-element-4'>Example X-ray Images</div>", unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>See what the app detects</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
        st.image(r"https://www.shutterstock.com/image-photo/blue-tone-radiograph-on-dark-600nw-2267523647.jpg", caption="Fractured X-ray")

with col2:
        st.image(r"https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpqwF54Bgvy8KwtTxT7W3mIEjH6MxqqMrYhg&s", caption="Normal X-ray")




# Footer
st.markdown("<div class='footer'>Trained on a dataset of X_ray images for fracture detection</div>", unsafe_allow_html=True)
