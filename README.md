**Project Description**


This project is aimed at detecting blood cancer from medical images using a Convolutional Neural Network (CNN). The code performs several steps, including data preprocessing, model training, and evaluation. Below is a step-by-step explanation of what each part of the code does.

Step-by-Step Explanation:-

1. Importing Libraries
The necessary libraries for image processing, data handling, machine learning, and deep learning are imported.

import pandas as pd 
import cv2                 
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

2. Setting Up Directories
Paths to the training and testing data directories are defined.

TRAIN_DIR = "C:/Users/User/Desktop/cancer/cancer/train/"
TEST_DIR = "C:/Users/User/Desktop/cancer/cancer/test/"

3. Preprocessing Function
Functions to preprocess the images are defined. These functions read images, resize them to 150x150 pixels, and convert them to arrays.

def get_label(Dir):
    for nextdir in os.listdir(Dir):
        if not nextdir.startswith('.'):
            if nextdir in ['NORMAL']:
                label = 0
            elif nextdir in ['CANCER']:
                label = 1
            else:
                label = 2
    return nextdir, label

def preprocessing_data(Dir):
    X = []
    y = []
    
    for nextdir in os.listdir(Dir):
        nextdir, label = get_label(Dir)
        temp = Dir + nextdir
        
        for image_filename in tqdm(os.listdir(temp)):
            path = os.path.join(temp + '/' , image_filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = skimage.transform.resize(img, (150, 150, 3))
                img = np.asarray(img)
                X.append(img)
                y.append(label)
            
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0
            elif nextDir in ['CANCER']:
                label = 1
            else:
                label = 2
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = skimage.transform.resize(img, (150, 150, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
                    
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

4. Loading and Preprocessing Data
The training and testing data are loaded and preprocessed using the previously defined functions.

X_train, y_train = get_data(TRAIN_DIR)
X_test, y_test = get_data(TEST_DIR)

5. Checking Data Shapes
The shapes of the preprocessed data arrays are printed to verify the dimensions.

print(X_train.shape, '\n', X_test.shape)
print(y_train.shape, '\n', y_test.shape)

6. One-Hot Encoding
The labels are converted to categorical format.

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
print(y_train.shape, '\n', y_test.shape)

7. Visualizing Data
A function is defined to visualize pairs of cancerous and normal images.

Pimages = os.listdir(TRAIN_DIR + "CANCER")
Nimages = os.listdir(TRAIN_DIR + "NORMAL")

def plotter(i):
    imagep1 = cv2.imread(TRAIN_DIR + "CANCER/" + Pimages[i])
    imagep1 = skimage.transform.resize(imagep1, (150, 150, 3), mode='reflect')
    imagen1 = cv2.imread(TRAIN_DIR + "NORMAL/" + Nimages[i])
    imagen1 = skimage.transform.resize(imagen1, (150, 150, 3))
    pair = np.concatenate((imagen1, imagep1), axis=1)
    print("(Left) - No CANCER Vs (Right) - CANCER")
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    plt.figure(figsize=(10, 5))
    plt.imshow(pair)
    plt.show()

for i in range(0, 5):
    plotter(i)

8. Model Definition
A Convolutional Neural Network (CNN) model is defined with several convolutional, activation, pooling, flatten, dense, and dropout layers.

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(150, 150, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.00005),
                  metrics=['accuracy'])

print(model.summary())

9. Model Training
The model is trained using the training data, with validation on the test data. Callbacks for learning rate reduction and model checkpointing are also used.

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

batch_size = 256
epochs = 10
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[lr_reduce, checkpoint], epochs=epochs)

10. Model Evaluation and Visualization
The training history is plotted to visualize accuracy and loss over epochs.

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

11. Confusion Matrix
A confusion matrix is generated to evaluate the model's performance on the test data.

pred = model.predict(X_test)
pred = np.argmax(pred, axis=1) 
y_true = np.argmax(y_test, axis=1)
CM = confusion_matrix(y_true, pred)
fig, ax = plot_confusion_matrix(conf_mat=CM, figsize=(5, 5))
plt.show()

**Summary**
This project demonstrates the application of deep learning techniques to medical image analysis for the detection of blood cancer. The code covers the entire workflow, from data preprocessing to model evaluation, and provides visualizations to help understand the model's performance.
