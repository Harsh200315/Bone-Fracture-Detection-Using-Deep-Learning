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
