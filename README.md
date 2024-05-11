# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset

- The problem at hand is the automatic classification of red blood cell images into two categories: parasitized and uninfected.
- Malaria-infected red blood cells, known as parasitized cells, contain the Plasmodium parasite
uninfected cells are healthy and free from the parasite.
- The goal is to build a convolutional neural network (CNN) model capable of accurately distinguishing between these two classes based on cell images.
- Traditional methods of malaria diagnosis involve manual inspection of blood smears by trained professionals, which can be time-consuming and error-prone.
- Automating this process using deep learning can significantly speed up diagnosis, reduce the workload on healthcare professionals, and improve the accuracy of detection.
- Our dataset comprises 27,558 cell images, evenly split between parasitized and uninfected cells.
- These images have been meticulously collected and annotated by medical experts, making them a reliable source for training and testing our deep neural network.

## Neural Network Model

![326722885-75400caf-48af-499d-aad1-5229fecd0ceb](https://github.com/RoopakCS/malaria-cell-recognition/assets/139228922/5d6d3636-6d94-4bea-a9fd-2012be7ba16d)


## DESIGN STEPS

### Step 1: 
We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.
### Step 2: 
To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.
### Step 3: 
We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.
### Step 4: 
We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.
### Step 5:
We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.
### Step 6: 
We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

## PROGRAM

### To share the GPU resources for multiple sessions
````python
pip install seaborn

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline

my_data_dir = 'dataset/cell_images'

os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)

len(os.listdir(train_path+'/uninfected/'))

len(os.listdir(train_path+'/parasitized/'))

os.listdir(train_path+'/parasitized')[0]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])

plt.imshow(para_img)

dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)

image_shape = (130,130,3)

help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')



train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


train_image_gen.class_indices

results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )

losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names

print("Vijayaraj V  212222230174")
model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)


print("Vijayaraj V  212222230174")
test_image_gen.classes

predictions = pred_probabilities > 0.5
print("Vijayaraj V  212222230174")
print(classification_report(test_image_gen.classes,predictions))


print("Vijayaraj V  212222230174")
confusion_matrix(test_image_gen.classes,predictions)

````

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/etjabajasphin/malaria-cell-recognition/assets/121303741/f1a4d9f0-bd69-4aec-9707-c81ef53dfbad)



### Classification Report

![image](https://github.com/etjabajasphin/malaria-cell-recognition/assets/121303741/eb6fb1b2-9e7c-4a2b-a941-915ecf890a14)



### Confusion Matrix

![image](https://github.com/etjabajasphin/malaria-cell-recognition/assets/121303741/22b9126a-6215-4812-b4b0-58580486b399)


### model

![image](https://github.com/etjabajasphin/malaria-cell-recognition/assets/121303741/234ece0b-339b-4056-a30b-20bfb21a849d)



## RESULT
Thus a deep neural network for Malaria infected cell recognition and to analyze the performance is created using tensorflow.
