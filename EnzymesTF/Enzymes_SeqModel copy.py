#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:54:05 2022

@author: michael.rowe
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
import time
from time import time
import matplotlib
from matplotlib import pyplot as plt
labels = np.loadtxt("Enzymes_labels.txt",dtype=int,ndmin=2)
features = np.loadtxt("Enzymes_features.txt")

def shuffle_tensors (features,labels):
    zipplis = []
    zipplis.clear()
    zipped = zip(features,labels)
    for el in zipped:
         zipplis.append(el)
    random.shuffle(zipplis)
    x,y = zip(*zipplis)
    x = np.array(x)
    y = np.array(y)
    return x,y


def labels_to_binary(labels):
    labels_binary = []
    for label in labels:
        if (label==1):
            labels_binary.append(np.array([1,0,0,0,0,0]))
            #np.append(labels_binary,np.array([1,0,0,0,0,0]), axis = 0)
        elif (label==2):
            #np.append(labels_binary,np.array([0,1,0,0,0,0 ]), axis = 0)
            labels_binary.append(np.array([0,1,0,0,0,0 ]))
        elif (label==3):
            #np.append(labels_binary,np.array([0,0,1,0,0,0 ]), axis = 0)
            labels_binary.append(np.array([0,0,1,0,0,0 ]))
        elif (label==4):
            #np.append(labels_binary,np.array([0,0,0,1,0,0 ]), axis = 0)
            labels_binary.append(np.array([0,0,0,1,0,0 ]))
        elif (label==5):
            #np.append(labels_binary,np.array([0,0,0,0,1,0 ]), axis = 0)
            labels_binary.append(np.array([0,0,0,0,1,0 ]))
        elif (label==6):
            #np.append(labels_binary,np.array([0,0,0,0,0,1]), axis = 0)
            labels_binary.append(np.array([0,0,0,0,0,1]))
    return labels_binary



x,OG_y = shuffle_tensors(features,labels)

OG_y_min_1 = []
for el in OG_y:
    OG_y_min_1.append(int(el)-1)
len(OG_y_min_1)


y = np.array(labels_to_binary(OG_y))



x_train = x[:480]
y_train = y[:480]
x_test = x[480:]
y_test = y[480:]

# for i in range(480):
#     print(x_train[i],y_train[i])

#class_labels = [6,5,1,2,3,4]


# ds =tf.data.Dataset.from_tensor_slices((features,labels))
# ds = ds.shuffle(10000).batch(2)

# ds_train = ds.take(round(600*0.8))
# ds_test = ds.skip(round(600*0.8))




# # cnn = models.Sequential([
# #           #cnn
# #         layers.Dense(2, activation="relu", name="layer1"),
# #         layers.Dense(3, activation="relu", name="layer2"),
# #         layers.Dense(4, name="layer3"),
# # ])

Enzyme_model = models.Sequential([
    layers.Dense(128, input_dim = 5, activation="relu", name="layer1"),
    layers.Dense(100, activation="relu", name="layer2"),
    layers.Dense(10, activation="softmax", name="layer3"),
    layers.Dense(6, activation="softmax", name="layer4"),
    
    
    
    #layers.Dense(7, name="layer3"),
          
      ])


Enzyme_model.compile(optimizer = 'sgd',
              #loss = 'binary_crossentropy',
              loss = 'mse',
              metrics = ['accuracy']
              #metrics=[tf.keras.metrics.BinaryCrossentropy()]
              )
              
 
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# file_name = 'my_saved_model'
# tensorboard = TensorBoard(log_dir ="logs//{}".format(file_name))

history = Enzyme_model.fit(x_train,y_train, batch_size = 1, epochs = 10, verbose = 1)

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
#plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





prediction = Enzyme_model.predict(x_train)    
def show_predictions(index):
    x = 0
    for i in range(index):  
        actual = np.where(y_train[i]==1)
        actlis = list(actual)
        Actual = int(actlis[0])
        Prediction = np.argmax(prediction[i])
        if (Prediction == Actual):
            
            print(x)
        x += 1

        
        # print('Actual: ' + f'{Actual}')
        # print('Prediction: ' + f'{Prediction}')
        # print("**************************************************************")

show_predictions(480)

print(OG_y[0]-1)
print(prediction[0])
print(np.argmax(prediction[0]))

#Making the Confusion Matrix
prediction = (prediction>.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.array(OG_y_min_1[:480]),prediction )
print(type(prediction))


