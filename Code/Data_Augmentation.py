#AUgmenatation des données !!
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from tqdm import tqdm 

from PIL import Image
import numpy as np
img_size=150
# Load an image using Pillow
def load_image(img_path):
    img = Image.open(img_path)  # Open the image
    img = img.convert('RGB')   # Ensure it's in RGB format
    img = img.resize((img_size,img_size))  # Resize the image to the target size
    return img

# Convert an image to a numpy array
def img_to_array(img):
    return np.array(img, dtype='float32')  # Convert to NumPy array
# Define paths for classes
class_0_path = '../Datasets/Train/Pneumonia'
class_1_path = '../Datasets/Train/Normal'

augmented_class_0_path = 'augmented_class_0_images'
augmented_class_1_path = 'augmented_class_1_images'

os.makedirs(augmented_class_0_path, exist_ok=True)
os.makedirs(augmented_class_1_path, exist_ok=True)
# Define the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
)

# Augment 20% of class 0
class_0_images = os.listdir(class_0_path)
np.random.shuffle(class_0_images)
subset_class_0 = class_0_images[:int(len(class_0_images) * 0.2)]
augmentations_per_image = 4  # Number of augmentations per image
for img_name in tqdm(subset_class_0,desc="Class 0 !"):
    img_path = os.path.join(class_0_path, img_name)
    img = load_image(img_path)  # Load the image using Pillow
    x = img_to_array(img)       # Convert to numpy array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    i=0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_class_0_path, save_prefix=f'aug_{os.path.splitext(img_name)[0]}', save_format='jpeg'):
        i+=1
        if i >=augmentations_per_image:
            break  # Save only 3 augmented image per original image




# Augment 100% of class 1 with multiple augmentations per image
class_1_images = os.listdir(class_1_path)
augmentations_per_image = 6  # Number of augmentations per image

for img_name in tqdm(class_1_images,desc="class 1"):
    img_path = os.path.join(class_1_path, img_name)
    img = load_image(img_path)  # Load the image using Pillow
    x = img_to_array(img)       # Convert to numpy array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_class_1_path, save_prefix=f'aug_{os.path.splitext(img_name)[0]}', save_format='jpeg'):
        i += 1
        if i >= augmentations_per_image:
            break  # Stop after generating the desired number of augmentations