# Bone Fracture Detection Using Deep Learning

An AI-powered bone fracture detection system that classifies X-ray images into **fractured** and **not fractured** categories using deep learning. The project includes a trained model, a Streamlit web app, and image-based prediction support.[3][1]

## Features

- Deep learning-based binary classification of X-ray images.
- Transfer learning with CNN architectures such as VGG16 and ResNet50.
- Streamlit web interface for real-time image upload and prediction.
- Custom UI styling with example X-ray visuals and error handling.
- Model loading using TensorFlow/Keras for inference.[2][1]

## Project Structure

```bash
Bone-Fracture-Detection-Using-Deep-Learning/
│
├── app.py
├── requirements.txt
├── model/
│   └── training.ipynb
└── data/
    └── images
```

## Requirements

Install the required dependencies first:

```bash
pip install tensorflow streamlit numpy pillow matplotlib
```

If you already have a `requirements.txt`, you can use:

```bash
pip install -r requirements.txt
```

## Step 1: Train the Model

1. Open `training.ipynb`.
2. Run all cells to preprocess the dataset, train the model, and save the final trained model.
3. Make sure the trained model is saved as:

```bash
model/recognition_model.keras
```

4. Confirm that the model file exists before running the app.

## Step 2: Run the Streamlit App

After training is complete and the model is saved, run the app using:

```bash
streamlit run app.py
```

If needed, you can also use:

```bash
python -m streamlit run app.py
```

## How It Works

1. Upload an X-ray image in JPG, JPEG, or PNG format.
2. The app preprocesses the image by resizing it to the expected input size.
3. The trained model predicts whether the image is **fractured** or **not fractured**.
4. The result is displayed instantly in the Streamlit interface.[1][2]

## Model Details

- Framework: TensorFlow / Keras
- UI: Streamlit
- Input type: X-ray image
- Output classes: fractured, not fractured
- Prediction type: Binary classification

## Example Usage

```bash
# Step 1: Train the model in training.ipynb
# Step 2: Start the app
streamlit run app.py
```

## Notes

- Make sure the model path in `app.py` matches the saved model location.
- Keep the trained `.keras` file in the correct folder before launching the app.
- Ensure image files and assets used in the UI are available in the repository.

## License

This project is for educational and research purposes.

***

## Small improvement I recommend
Your current `app.py` uses an absolute local path for the model:

```python
model = load_model("/Users/harshsoni2003/Downloads/Final/model/recognition_model.keras")
```

For GitHub and portability, change it to a relative path like:

```python
model = load_model("model/recognition_model.keras")
```

That will make your README steps work properly on any system.

If you want, I can also give you:
- a **proper requirements.txt**
- a **shorter professional README**
- or a **GitHub-ready README with badges and screenshots section**

Sources
[1] Run your Streamlit app https://docs.streamlit.io/develop/concepts/architecture/run-your-app
[2] Whole model saving & loading - Keras https://keras.io/api/models/model_saving_apis/model_saving_and_loading/
[3] Streamlit • A faster way to build and share data apps https://streamlit.io
[4] Install Streamlit https://docs.streamlit.io/get-started/installation
[5] Streamlit - Complete Setup Guide https://www.geeksforgeeks.org/python/streamlit-introduction-and-setup/
[6] Step by Step Running | PDF https://id.scribd.com/document/918674495/Step-by-Step-Running
[7] Import trained LSTM model on Streamlit https://stackoverflow.com/questions/78254579/import-trained-lstm-model-on-streamlit
[8] streamlit/README.md at develop · streamlit/streamlit https://github.com/streamlit/streamlit/blob/develop/README.md
[9] Install Streamlit using command line https://docs.streamlit.io/get-started/installation/command-line
[10] GitHub - cedanl/streamlit-app-template: ✨ [NEW] - A production-ready template for building & deploying Streamlit apps with best practices and essential configurations. ⚡ Runs with uv https://github.com/cedanl/streamlit-app-template
[11] Streamlit https://pypi.org/project/streamlit/
[12] Create a Web App from Your TensorFlow Neural Network and Embed it in Medium.com https://drlee.io/create-a-web-app-from-your-tensorflow-neural-network-and-embedding-it-in-medium-com-4b4b8501cccc?gi=c558f586ac3e
[13] GitHub - OmarAlkousa/Learn-Streamlit: This repository is to represent how easy to design a web app using the Streamlit package. The best way to learn is to practice... https://github.com/OmarAlkousa/Learn-Streamlit
[14] Loading Keras models - Using Streamlit https://discuss.streamlit.io/t/loading-keras-models/7289
[15] Unable to load my saved model using tensorflow keras https://discuss.streamlit.io/t/unable-to-load-my-saved-model-using-tensorflow-keras/13026
# Bone-Fracture-Detection-Using-Deep-Learning
