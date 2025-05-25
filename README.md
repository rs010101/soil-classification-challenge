Soil Classification with ResNet-18

Project Overview
This project performs soil type classification using images. A pretrained ResNet-18 model is fine-tuned to classify soil images into different soil types based on a labeled dataset. The trained model can then be used to predict soil types on unseen test images.

Dataset
Train Images: Located in the train folder.
Test Images: Located in the test folder.
Labels: Provided in train_labels.csv (with image IDs and soil types).
Test IDs: Provided in test_ids.csv (image IDs only).

Installation
Clone the repository:
git clone https://github.com/yourusername/soil-classification.git
cd soil-classification

Install dependencies:
pip install -r requirements.txt

Usage
Training
Run the training.ipynb notebook to train the ResNet-18 model:
Loads and preprocesses images and labels.
Fine-tunes ResNet-18 for 5 epochs.
Saves the trained model weights and label encoder.

Inference & Validation
Run the inference.ipynb notebook to:
Load the saved model and label encoder.
Evaluate model performance on the validation set.
Predict soil types on test images.

Model Details
Architecture: ResNet-18 (pretrained on ImageNet).
Input: 224x224 RGB images.
Output: Number of soil classes.
Loss: Cross-Entropy Loss.
Optimizer: Adam (learning rate = 1e-4).
Metrics: Weighted F1 score and classification report.

Results
Validation weighted F1 score: 0.97.
Classification report available after validation.

Future Work
Experiment with data augmentation.
Use deeper or ensemble models.
Implement learning rate scheduling and early stopping.
Increase training epochs for better accuracy.
Add automated hyperparameter tuning.

Acknowledgments
Dataset and challenge provided by https://www.kaggle.com/competitions/soil-classification 

