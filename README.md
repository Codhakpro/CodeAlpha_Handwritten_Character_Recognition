# Handwritten Character Recognition

A machine learning project that uses a Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits from the MNIST dataset.

This project was completed as part of the CodeAlpha Machine Learning Internship.

## Project Overview

The goal of this project is to train a CNN model to classify handwritten digits from **0 to 9**.

The model is trained using the MNIST dataset, which contains thousands of handwritten digit images. After training, the model is evaluated on unseen test data and can also be used to make predictions on a custom handwritten digit image.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Dataset

The project uses the **MNIST handwritten digit dataset**.

The dataset contains:

- 60,000 training images
- 10,000 testing images
- 10 digit classes (0–9)
- Images with a size of 28 × 28 pixels

The dataset is automatically downloaded through Torchvision when the notebook is executed.

## Model

A Convolutional Neural Network (CNN) was developed using PyTorch.

The network consists of:

- Convolutional layers
- ReLU activation functions
- Max-pooling layers
- Fully connected layers
- Dropout
- A final output layer for the 10 digit classes

The model was trained using:

- Loss function: Cross Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Training epochs: 2

## Model Evaluation

After training, the model was evaluated using the MNIST test dataset.

The project includes:

- Test accuracy
- Sample predictions
- Confusion matrix

The confusion matrix helps show how well the model distinguishes between the different digit classes.

## Custom Handwritten Digit Test

In addition to the MNIST test dataset, the trained model was tested using a custom handwritten digit image.

The project includes the handwritten test image:

`second-image.jpg`

The image is processed and passed to the trained CNN model to generate a prediction.

> **NOTE:** For the custom image test, the handwritten digit should be **bold, clear, and have thick strokes**. The model was trained on MNIST-style images, so very thin, faint, or unclear handwriting may result in incorrect predictions.

## Project Files

```text
CodeAlpha_Handwritten_Character_Recognition/
│
├── data/
│   └── MNIST dataset files
│
├── handwritten_character_recognition.ipynb
├── handwritten_character_cnn.pth
├── second-image.jpg
├── results.txt
├── requirements.txt
└── README.md
```
# How to Run
1. Install the required packages
``` 
pip install -r requirements.txt
```
2. Open the Jupyter Notebook
```
jupyter notebook
```
Open:
```
handwritten_character_recognition.ipynb
```
3. Run the notebook
```
Run the cells in order to:
```
1. Load the MNIST dataset. 
2. Prepare the data. 
3. Build the CNN. 
4. Train the model. 
5. Evaluate the model. 
6. Generate predictions. 
7. Save the trained model.

4. Test a Custom Handwritten Digit. 

The trained model can also be tested using the custom image included in the project:
```
second-image.jpg
```
The image is loaded, converted to grayscale, inverted, resized to 28 × 28 pixels, and passed to the trained CNN model for prediction.

# Results

The trained CNN successfully learned to recognize handwritten digits from the MNIST dataset.

The project also demonstrates how a trained model can be used to make predictions on a custom handwritten digit image.

# Conclusion

This project demonstrates a complete image classification workflow using a Convolutional Neural Network and PyTorch. It covers dataset preparation, model development, training, evaluation, visualization, model saving, and testing the trained model with a custom handwritten image.

# CodeAlpha Internship

This project was completed as part of the CodeAlpha Machine Learning Internship.

Task: Handwritten Character Recognition
