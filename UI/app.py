# app.py (Python backend)
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
from skimage.feature import hog
import cv2  # Assuming you're using OpenCV for image loading
import joblib  # For loading your trained model
import cv2


from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)
#************************* Fourier *********************************
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
            local_features.append(normalized_patch_features)  # Flatten and store  features
    return np.concatenate(local_features)  # Concatenate  features



#**************************** Gabor *****************************
def gabor(image, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5):
    filters = []
    for theta in np.arange(0, np.pi / 2, np.pi/4):
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
                features.extend(normalized_patch_features)  # Collect all features
    return np.array(features)  # Combine features



#********************* DCT *********************************

def dct_(image, window_size=8, top_k=5):
    h, w = image.shape
    local_features = []
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            patch = image[y:y + window_size, x:x + window_size]
            dct_patch = cv2.dct(np.float32(patch) / 255.0)
            dct_flat = dct_patch.flatten()
            normalized_patch_features = normalize(dct_flat[:top_k])  # Normalize l DCT features
            local_features.append(normalized_patch_features)
    return np.array(local_features)  # Return as numpy array


#******************************************************************************************************
def phog_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  # Assuming square-shaped input
        image = image.reshape((size, size))
    features = phog(image)
   
    return features

def gabor_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  
        image = image.reshape((size, size))
    features = gabor(image)
  
  
    return features

def fourier_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  
        image = image.reshape((size, size))
    global_features = fourier(image)
    
    
    return global_features

def dct_features(image):
    if image.ndim == 1:
        size = int(np.sqrt(len(image)))  
        image = image.reshape((size, size))
    features = dct_(image).flatten()
    
    return features  # Final result (1D array)
#******************************************************************************************************

def normalize(array):
    min_val = np.min(array)
    maX_test = np.max(array)
    normalized = (array - min_val) / (maX_test - min_val + 1e-8)  # Avoid division by zero
    return normalized


# Load images from the directory
def load_images(file,imgsize,model_type):
    images=[]
    if file.filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other formats if needed
        # Read the file content into memory
        file_content = file.read()
        # Decode the image from the byte stream (in grayscale)
        image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Resize the image
        resized = cv2.resize(image, (imgsize, imgsize))
        
        
        # Append to images list for ML
        images.append({'name': file.filename, 'image': resized})
        if model_type=='ML':

            return images
        elif model_type =='DL':
            image = resized / 255.0  # normalize to [0,1]
            image=image.astype('float32')
            print('shape: ',image.shape)
            image = image.reshape(-1, imgsize, imgsize, 1)  # reshape for the model
            return image,resized
        

# Predict function
def predict_images(model_path, file, classes,imgsize,model_type):
    # Load the trained model
    model = joblib.load(model_path)

    # Load new images
    image_data = load_images(file,imgsize,model_type)

    predictions = []
    for entry in image_data:
       # Combine global and local features
        combined_features = np.concatenate([
            phog_features(entry['image']),
          
            gabor_features(entry['image']),
           
            fourier_features(entry['image']), 
        
            dct_features(entry['image']), 
            
        ])
         # Convert to float32 for consistency with model requirements
        combined_features = np.array(combined_features,dtype=np.float32)
        # Reshape combined features to match model input
        combined_features = combined_features.reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(combined_features)
        probabilities = model.predict_proba(combined_features) #get proba like [0.44 , 0.56]
        print('prediction',prediction)
        predictions.append({
            "image_name": entry['name'],
            
            "predicted_label": classes[prediction[0]]  # Convert label index to class name
        })

    return predictions,probabilities,image_data
#after app.route : all the function called automaticly , but before the app.route , it is our dicision to call the fucntion ot not !
#Flask will only call predict_mask() when the /predict-pnuemonia endpoint receives a POST request.!!!! , if no post request = no read !!!!
@app.route('/predict-pneumonia', methods=['POST'])
def predict_pneumonia():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        # Get the additional variable
        model_type = request.form.get('model')  # Use get to avoid KeyError
        if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
        # Specify the directory for new images and model path
        if model_type=='ML':
            img_size=150
            model_path = "../Models/ML_Concatinated.pkl"  # Change to your model's path
            classes = ["PNEUMONIA","NORMAL" ]  # Adjust according to your labels

            # Predict the labels of new images
            predicted_results,probability,image_data = predict_images(model_path, file, classes,img_size,model_type) #model_type c'est just DL ou ML !!

            # Save the image after prediction
            img_used_dir = "images_used"
            os.makedirs(img_used_dir, exist_ok=True)  # Create directory if it doesn't exist
            for i, result in enumerate(predicted_results):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = 'P' if result['predicted_label'] == 'PNEUMONIA' else 'N'
                new_filename = f"{prefix}_{timestamp}_ML.png"  # You can change the format as needed
                temp_image_path = os.path.join(img_used_dir, new_filename)
                cv2.imwrite(temp_image_path, image_data[i]['image'])  # Save the resized image

            # Print the predictions
            for result in predicted_results:
                #print(f"Image: {result['image_name']}, Predicted Label: {result['predicted_label']}")
                return jsonify({"result": result['predicted_label'], "confidence": float(probability[0, 1])})
        elif model_type=='DL':
            # Load the trained model
            model = load_model("../Models/Model_DL.h5")

            # Image size used during training
            img_size = 150

            # Define labels
            labels = ['PNEUMONIA', 'NORMAL']

            """Load and preprocess a single image."""
            img,resized =load_images(file,img_size,model_type)
            prediction = model.predict(img)
            proba=prediction[0][0]
            class_index = int(prediction[0][0] > 0.5)  # Threshold at 0.5 for binary classification
            prediction = labels[class_index]
            
            # Save the image after prediction
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = 'P' if prediction == 'PNEUMONIA' else 'N'
            new_filename = f"{prefix}_{timestamp}_DL.png"  # You can change the format as needed
            img_used_dir = "images_used"
            os.makedirs(img_used_dir, exist_ok=True)  # Create directory if it doesn't exist
            temp_image_path = os.path.join(img_used_dir, new_filename)
            cv2.imwrite(temp_image_path, resized)  # Save the resized image

            return jsonify({"result": prediction,"confidence":float(proba),})
        else:
            return jsonify({"error": "Invalid model type"}), 400
       
        
    except Exception as e:
        print("Error occurred:", str(e))  # Log the error
        return jsonify({"error": "An error occurred: " + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
