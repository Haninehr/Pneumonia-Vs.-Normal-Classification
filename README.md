# Pneumonia vs. Normal Classification
![Pneumonia Icon](UI/icon-logo.png) 

**Author:** Rouibah Hanine  <br>
 

This project performs **binary classification** of chest X-ray images: **Pneumonia** vs **Normal**.<br>

Two main approaches were developed:<br>

1. Direct training of a **Convolutional Neural Network (CNN)** on images  <br>
2. Hand-crafted feature extraction (**PHOG**, **Gabor**, **Fourier**, **DCT**) using two strategies:  <br>
   - Sequential processing  <br>
   - Concatenated features  <br>
   followed by classification with **SGDClassifier** (logistic loss) <br>


## Structure du projet

Pneumonia Vs. Normal Classification/<br>
├──Code/<br>
    ├────── CNN.py                        # Train CNN directly<br>
    ├────── Data_Augmentation.py          # Script d'augmentation<br>
    ├────── FeaturesExtraction_Sequentiel.py # Sequential Features Extraction<br>
    ├────── FeaturesExtraction_Concatenated.py # Concatenated Features Extraction<br>
    ├────── Train_Sequentiel_Features.py # Train with Sequential Features<br>
    ├────── Train_Concatenated_Features.py # Train with Concatenated Features<br>
├──Data # Extarcted Features<br>
├──Datasets<br>
├──Models <br>
├──UI   #Interface of prediction<br>
├──Report<br>

## Prediction Interface
![APP screenshot](Screenshots/UI.png)
<br>
In addition to the models, a **user-friendly web interface** was developed for real-time prediction.<br>

### Features
- Select model: **CNN** or **ML (Concatenated Features)**<br>
- Upload a chest X-ray image<br>
- Instant result: **Sick** (Pneumonia) or **Not Sick** (Normal)<br>
- Confidence percentage<br>
- Automatically saves the uploaded image with prediction<br>
### requirements
-Flask<br>

### How To Run 
->py -m flask run --port 8000 <br>

## Report
### Key facts from the report

-Training set: 3,200 Pneumonia + 708 Normal<br>
-Validation set: 651 Pneumonia + 621 Normal<br>
-Strong initial class imbalance → addressed with targeted data augmentation<br>
-Augmentation rules:<br>
    -20% of Pneumonia images<br>
    -100% of Normal images<br>
-Applied transformations: rotation ±15°, shift 5%, zoom 10%, horizontal flip<br>
📄 [Full Report (PDF)](Report/Report.pdf)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
