#L'extraction des features :Sequentiel ::: Gabor => Fourier => DCT => Phog !
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg') #use this to avoid problem socket !!
import os

import pandas as pd
import matplotlib.pyplot as plt

from collections import deque

data_val='../Data/val_data_Sequentiel.csv'
data_train='../Data/train_data_Sequentiel.csv'
#********************* Fourier *******************************
def fourier(image, window_size=32):
    h, w = image.shape
    local_features = []
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            patch = image[y:y + window_size, x:x + window_size]
            f_transform = np.fft.fft2(patch)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # Avoid log(0)
            normalized_patch_features = normalize(magnitude_spectrum.flatten())  # Normalize patch features
            local_features.append(normalized_patch_features)  # Flatten and store features
    return np.concatenate(local_features)  # Concatenate 



#**************************** Gabor *****************************
def gabor(image, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5):
    filters = []
    for theta in np.arange(0, np.pi / 2 , np.pi/4):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(image, cv2.CV_32F, kern)
        normalized_filtered_img = normalize(filtered_img.flatten())  # Normalize and flatten
        filters.append(normalized_filtered_img)  # Append normalized features
    return np.concatenate(filters)  # Combine all filtered images into one vector


#****************** pHOG ***************************************

def phog(image, orientations=6, pixels_per_cell=(4, 4), cells_per_block=(1, 1), levels=2): #levels = 3!
    features = []
    for level in range(levels):
        cell_size = max(16, 2 ** level)
        for y in range(0, image.shape[0], cell_size):
            for x in range(0, image.shape[1], cell_size):
                patch = image[y:y + cell_size, x:x + cell_size]
                hog_features = hog(patch, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                   cells_per_block=cells_per_block, block_norm='L2-Hys')
                normalized_patch_features = normalize(hog_features)
                features.extend(normalized_patch_features)  # Collect all  features
    return np.array(features)  # Combine  features



#********************* DCT *********************************

def dct_(image, window_size=8, top_k=5):
    h, w = image.shape
    local_features = []
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            patch = image[y:y + window_size, x:x + window_size]
            dct_patch = cv2.dct(np.float32(patch) / 255.0)
            dct_flat = dct_patch.flatten()
            normalized_patch_features = normalize(dct_flat[:top_k])  # Normalize  DCT features
            local_features.append(normalized_patch_features)
    return np.array(local_features, dtype=object)  # Return as numpy array


#******************************************************************************************************
def phog_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  # Assuming square-shaped input
        dif= len(image)-(size*size)
        if dif==0:
            image = image.reshape((size, size))
        else:
            size+=1
            image = pad_1d_to_square(image, target_size=size)
    features = phog(image)
   
    return features

def gabor_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  
        dif= len(image)-(size*size)
        if dif==0:
            image = image.reshape((size, size))
        else:
            size+=1
            image = pad_1d_to_square(image, target_size=size)
    features = gabor(image)
  
  
    return features

def fourier_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  
        dif= len(image)-(size*size)
        if dif==0:
            image = image.reshape((size, size))
        else:
            size+=1
            image = pad_1d_to_square(image, target_size=size)
            
    global_features = fourier(image)
    
    
    return global_features

def dct_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  
        dif= len(image)-(size*size)
        if dif==0:
            image = image.reshape((size, size))
        else:
            size+=1
            image = pad_1d_to_square(image, target_size=size)
    features = dct_(image).flatten()
    
    return features  # Final result (1D array)
#******************************************************************************************************
def normalize(array):
    min_val = np.min(array)
    maX_val = np.max(array)
    normalized = (array - min_val) / (maX_val - min_val + 1e-8)  # Avoid division by zero
    return normalized

def pad_1d_to_square(array, target_size=0):
    """
    Pads the 1D array to form a square of size target_size x target_size.
    """
    current_size = len(array)
    target_length = target_size * target_size  # Calculate the size of the target square
    
    # Calculate how many elements we need to pad
    pad_length = target_length - current_size
    
    # Pad with zeros (you can change this to other padding strategies if needed)
    padded_array = np.pad(array, (0, pad_length), mode='constant', constant_values=0)
    
    # Reshape into a square matrix
    padded_matrix = padded_array.reshape(target_size, target_size)
    
    return padded_matrix



def extract_combined_features(image):
    gabor_result = gabor_features(image)
    fourier_result = fourier_features(gabor_result)
    dct_result =dct_features(fourier_result)
    final_features   =phog_features(dct_result)
    final_features = [
    item if isinstance(item, np.ndarray) else np.array([item]) 
    for item in final_features
    ]
    final_features=np.concatenate(final_features)
    
    return final_features

# Path to your dataset directory
data_train_dir = "../Datasets/Train"
data_val_dir = "../Datasets/Val"
print("====== Start ======")
classes = ['PNEUMONIA', 'NORMAL']

#

img_size=150
# Function to load images and their labels
def load_images(data_dir, classes):
   
    image_data = []
    
    for label, category in enumerate(classes):
        category_path = os.path.join(data_dir, category)
        for img_name in tqdm(os.listdir(category_path), desc=f"Loading images for {category}", leave=False):
            img_path = os.path.join(category_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Load image in grayscale
            # Resize the image
            resized_image = cv2.resize(image, (img_size,img_size))
            image_data.append({
                "name": img_name,
                "label": label,
                "image": resized_image
            })
            
    # Shuffle the image data to randomize
    #random.shuffle(image_data)
    
    return image_data

def extract_val_images_features(data_val_dir):
    while True:
        break
        yes_or_no = input(f"Do you want rename the validation files? Type 'yes' to rename or 'no' to skip to features extraction! ").strip().lower()
        if yes_or_no == "yes":
            process_subfolders(data_val_dir)
            break
        elif yes_or_no == "no":
            print('skipping !')
            break
        else:
            print("Invalid choice. Please type 'yes' or 'no'.")
    # Load all images and their labels
    print("start loading valing images !")
    val_image_data = load_images(data_val_dir, classes) 
    print("End loading valing images !")
    features_data=[]
    labels=[]
    img_name=[]
    
    # Convert list to deque for efficient popleft operations
    val_image_data = deque(val_image_data)
    print('Start extracting features from validing image !')
    # Apply methods and extract features for each image
    for _ in tqdm(range(len(val_image_data)), desc="Extracting features from validing images"):
        entry = val_image_data.popleft()  # Remove the first entry efficiently
        combined_features = extract_combined_features(entry['image'])
       
        features_data.append(combined_features)
        labels.append(entry['label'])  # Collect the labels
        img_name.append(entry['name'])
        del combined_features
    print('End extracting validation image features !')
    print('Saving Data ... !')
    # Create a DataFrame
    data = {
        "img_name":np.array(img_name),
        "labels": np.array(labels),
        "features": [",".join(map(str, row)) for row in np.array(features_data, dtype=np.float32)]
    }
    df = pd.DataFrame(data)
    # Save to CSV
    df.to_csv(data_val, index=False)
    df=0
    features_data=[]
    labels=[]
    img_name=[]
 
    

    print("val CSV file saved successfully!")

def extract_train_images_features(data_train_dir):
    while True:
        break
        yes_or_no = input(f"Do you want rename the training files? Type 'yes' to rename or 'no' to skip to features extraction! ").strip().lower()
        if yes_or_no == "yes":
            process_subfolders(data_train_dir)
            break
        elif yes_or_no == "no":
            print('skipping !')
            break
        else:
            print("Invalid choice. Please type 'yes' or 'no'.")

    # Load all images and their labels
    print("start loading training images !")
    # Step 1: Load images
    train_image_data = load_images(data_train_dir, classes)    
    
    print('start extarcting features !')
    # Create a list to store features and labels
    features_data = []
    labels = []
    img_name=[]
    # Convert list to deque for efficient popleft operations
    train_image_data = deque(train_image_data)

   
    for _ in tqdm(range(len(train_image_data)), desc="Extracting features from training images!"):
        entry = train_image_data.popleft()  # Remove the first entry efficiently

        combined_features=extract_combined_features(entry['image'])
        #print('image name :',entry['name'])
        features_data.append(combined_features)
        del combined_features
        labels.append(entry['label'])  # Collect the labels
        img_name.append(entry['name'])
    print("Saving Data ... !")
    # Create a DataFrame
    data = {
    "img_name":np.array(img_name),
    "labels": np.array(labels),
    "features": [",".join(map(str, row)) for row in np.array(features_data, dtype=np.float32)]
    }
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(data_train, index=False)
    print("Train CSV file saved successfully!")
    features_data = []
    
    labels=[]
    img_name=[]
    
    print('End extarcting features from  training images !')



extract_val_images_features(data_val_dir)
extract_train_images_features(data_train_dir)


