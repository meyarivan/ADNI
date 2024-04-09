# Alzheimer's Disease Classification Using Convolutional Neural Networks (CNN)

In this project, we aim to classify brain MRI images into different categories related to Alzheimer's disease using Convolutional Neural Networks (CNN). We'll be leveraging image data from the Alzheimer's Dataset.

## Setup

To run this project, you can follow these steps:

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Introduction

Alzheimer's disease is a neurodegenerative disorder that affects millions of people worldwide. Early detection and classification of Alzheimer's disease from brain MRI images are crucial for effective treatment and management.

## Data Preprocessing

We start by loading brain MRI images from the Alzheimer's Dataset and preprocessing them. The dataset is organized into different folders representing different classes: Alzheimer's disease, Cognitively normal, Early mild cognitive impairment, and Late mild cognitive impairment.

## Data Exploration

We explore the dataset by visualizing sample images and analyzing the distribution of samples across different classes.

## Data Augmentation

We perform data augmentation using ImageDataGenerator to generate additional training samples and prevent overfitting.

## Model Building

We build a CNN model for classifying brain MRI images. The model consists of multiple convolutional layers followed by max-pooling layers, dropout layers for regularization, and dense layers for classification. Additionally, we incorporate the InceptionV3 pre-trained model for feature extraction.

## Model Training

We train the CNN model using the augmented training data and validate it using a validation set. We monitor the training process using callbacks like EarlyStopping and ModelCheckpoint to prevent overfitting and save the best model.

## Model Evaluation

We evaluate the trained model on the test set and analyze its performance using metrics like accuracy, loss, confusion matrix, and classification report.

## Results

The trained CNN model, including the InceptionV3 pre-trained model, achieves high accuracy in classifying brain MRI images into different categories related to Alzheimer's disease. It shows promising results for early detection and classification of Alzheimer's disease from brain MRI images.

## Usage

### Running the Jupyter Notebook

1. Clone or download this repository to your local machine.

```bash
git clone <repository_url>
```

2. Navigate to the project directory.

```bash
cd <repository_name>
```

3. Launch Jupyter Notebook.

```bash
jupyter notebook
```

4. In your browser, open the Jupyter Notebook file `Alzheimers_Prediction_Project.ipynb`.

5. Follow the instructions in the notebook to execute each cell and run the code.

### Additional Notes

- Ensure you have the necessary dataset files available in the specified directory or update the paths in the notebook accordingly.
- Make sure to adjust any file paths or configurations as needed based on your local setup.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust any part of the README according to your preferences or add more sections as needed!
