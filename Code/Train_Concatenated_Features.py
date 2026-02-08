#Train : Concatenation des 4 method [PHOG,Gabor,Fourier,DCt] !!
#Rouibah Hanine 
#started : 2024-12-14 18:19:22 !

import matplotlib
matplotlib.use('TkAgg') #use this to avoid problem socket !!
import numpy as np
import joblib
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Import time module
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,log_loss
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import SGDClassifier
# Set NumPy print options to display all elements
np.set_printoptions(threshold=np.inf)  # Show all elements

print("====== Start ======")
classes = ['PNEUMONIA', 'NORMAL']
data_train_extracted="../Data/train_data_Concatinated.csv"
data_val_extracted="../Data/val_data_Concatinated.csv"


# Load features when needed
df = pd.read_csv(data_train_extracted)
# Extract labels
Y_train = df['labels'].values
# Extract features and convert to numpy array
X_train = np.array([np.fromstring(features, sep=',') for features in df['features']])


df = pd.read_csv(data_val_extracted)
# Extract labels
Y_val = df['labels'].values
# Extract features and convert to numpy array
X_val = np.array([np.fromstring(features, sep=',') for features in df['features']])

        
# Initialize logistic regression model with gradient descent solver
#model = LogisticRegression(solver='lbfgs', max_iter=1000)  # lbfgs is a gradient descent-based solver
#switching from LogisticRegression (which processes all data at once) to SGDClassifier with loss='log' 
# (which uses mini-batches) should help with memory allocation issues, especially for large datasets.
start_time = time.time()  # Start timer

# Initialize the model
model = SGDClassifier(loss='log_loss', 
                    max_iter=1, 
                    tol=None, 
                    random_state=23, 
                    warm_start=True,
                    alpha=0.0001,
                     eta0=0.0001,#learning rate
                    learning_rate='adaptive',
                    shuffle=True,
                     penalty='elasticnet',  # Elastic Net regularization
                    l1_ratio=0.5,  # 50% L1 and 50% L2 !!
                   
                    )

# Parameters for early stopping
n_epochs = 1000  # Maximum number of epochs
patience = 20  # Number of epochs to wait for improvement
best_val_accuracy = 0
epochs_without_improvement = 0


# Metrics to track
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
# Training with early stopping
for epoch in range(n_epochs):
    model.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
    
    # Training accuracy and loss
    Y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    train_loss = log_loss(Y_train, model.predict_proba(X_train))
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)
    
    # Validation accuracy and loss
    Y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(Y_val, Y_val_pred)
    val_loss = log_loss(Y_val, model.predict_proba(X_val))
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Early stopping logic
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0
        joblib.dump(model, "../Models/ML_Concatinated.pkl")  # Save the current best model
        print(f"Best model saved at epoch {epoch + 1}.")
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break
X_train=0
Y_train=0
end_time = time.time()  # End timer
# Calculate execution time
execution_time = end_time - start_time
print(f"Execution Time for Training: {execution_time:.4f} seconds")

# Plot Training and Validation Accuracy and Loss
epochs = range(1, len(train_accuracies) + 1)

fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_accuracies , 'go-' , label = 'Training Accuracy',markersize=2)
ax[0].plot(epochs , val_accuracies , 'ro-' , label = 'Validation Accuracy',markersize=2)
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_losses , 'g-o' , label = 'Training Loss',markersize=4)
ax[1].plot(epochs , val_losses , 'r-o' , label = 'Validation Loss',markersize=4)
ax[1].set_title('valing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()
model = joblib.load("../Models/ML_Concatinated.pkl")
# Make predictions on the val set
Y_pred = model.predict(X_val)
y_pred_proba=model.predict_proba(X_val)
# Print classification report
print(classification_report(Y_val, Y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(Y_val, Y_pred)
# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['PNEUMONIA', 'NORMAL'], yticklabels=['PNEUMONIA', 'NORMAL'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(Y_val, Y_pred)
# Calculate AUC
auc = roc_auc_score(Y_val, Y_pred)
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
#print('y_pred : ',y_pred_proba)


print("======= End =======")

#End !