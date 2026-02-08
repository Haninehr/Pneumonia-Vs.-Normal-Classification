#CNN !!
#Rouibah Hanine 
#started : 2024-12-14 18:04:26 !

import numpy as np 
import pandas as pd 
import time  # Import time module

#Critère de comparaison : Accuracy and execution time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
#from keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
import cv2
import os



labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in tqdm(os.listdir(path), desc=f"Loading images ", leave=False):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    random.shuffle(data)
    return np.array(data,dtype=object)


print("start loading training images !")
train_images = get_training_data('../Datasets/Train')
print("End loading training images !")




X_train = []
Y_train = []


for feature, label in train_images:
    X_train.append(feature)
    Y_train.append(label)




print("start loading Validating images !")
test_images = get_training_data('../Datasets/Val')
print("End loading Validating images !")

X_test = []
Y_test = []


for feature, label in test_images:
    X_test.append(feature)
    Y_test.append(label)



# Normalize the data
X_train = np.array(X_train,dtype=object) / 255

X_test = np.array(X_test,dtype=object) / 255

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# resize data for deep learning 
X_train = X_train.reshape(-1, img_size, img_size, 1)
Y_train = np.array(Y_train)

X_test = X_test.reshape(-1, img_size, img_size, 1)
Y_test = np.array(Y_test)



model = Sequential() 
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (img_size,img_size,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))

#Optimizer : rmsprop !
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy']) 

model.summary()

# Measure the execution time for model training
start_time = time.time()  # Start timer
epochs = 15
#Reduce LR on Plateau !
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=3,factor=0.3, min_lr=0.00000001)
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # 
    patience= 3 ,          # Number of epochs to wait without improvement !
    restore_best_weights=True  # Restore the best weights after stopping!!
)

history = model.fit(X_train,Y_train, batch_size = 32 ,epochs = epochs , validation_data = (X_test, Y_test) 
                    ,callbacks = [learning_rate_reduction,early_stopping])

end_time = time.time()  # End timer ..

# Calculate execution time
execution_time = end_time - start_time
print(f"Execution Time for Training: {execution_time:.4f} seconds")

print("Loss of the model is - " , model.evaluate(X_test,Y_test)[0])
print("Accuracy of the model is - " , model.evaluate(X_test,Y_test)[1]*100 , "%")

model.save("../Models/Model_DL.h5")
print('model saved !')

#Analysis after Model Training !
epochs = [i for i in range(len(history.history['loss']))]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()


# Make predictions
predictions = model.predict(X_test) 

# Convert probabilities to binary labels (0 or 1)
predictions = (predictions > 0.5).astype(int)  # Apply threshold of 0.5 to get binary labels

# Print the classification report
print(classification_report(Y_test, predictions, target_names=['PNEUMONIA', 'NORMAL']))

# Compute confusion matrix
cm = confusion_matrix(Y_test, predictions)

# Convert confusion matrix to DataFrame for better labeling in heatmap
cm_df = pd.DataFrame(cm, index=['PNEUMONIA', 'NORMAL'], columns=['PNEUMONIA', 'NORMAL'])

# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


from sklearn.metrics import roc_curve, roc_auc_score

# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(Y_test, predictions)

# Calculate AUC
auc = roc_auc_score(Y_test, predictions)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#ENd !!