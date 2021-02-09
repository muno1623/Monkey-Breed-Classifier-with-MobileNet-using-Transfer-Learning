import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow.keras.utils as np_utils
import tensorflow.keras.utils as np_utils
from tensorflow.keras.models import load_model
import pydot
import os
from os import listdir
from os.path import isfile, join
import sys
import shutil
import pickle
from tensorflow.keras.models import Model

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224

#Loading Data
train_data_dir = './monkey_breed/train'
validation_data_dir = './monkey_breed/validation'

batch_size = 16

# Let's use some data augmentaiton
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# set our batch size (typically on most mid tier systems we'll use 16-32)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights='imagenet',
                      include_top=False,
                      input_shape=(img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False

# Let's print our layers
for (i, layer) in enumerate(MobileNet.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)

def addTopModelMobileNet(bottom_model, num_classes):
    """creates the top or head of the model that will be
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model

# Set our class number to 3 (Young, Middle, Old)
num_classes = 10

FC_Head = addTopModelMobileNet(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())



#Creating Callbacks

# Checkpoint, Early Stopping, and Learning rates
checkpoint = ModelCheckpoint("E:/Computer Vision and Machine Learning Projects/BuildingCNN/Trained Models/Fruit_Classifier_Checkpoint.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 3, #Number of epochs we wait before stopping
                          verbose = 1,
                          restore_best_weights = True) #keeps the best weigths once stopped

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 3, verbose = 1, min_delta = 0.001)

callbacks = [checkpoint, earlystop, reduce_lr]

# We use a very small learning rate
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 1098
nb_validation_samples = 272

# We only train 5 EPOCHS
epochs = 5
batch_size = 16

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

#Visualizing Model
np_utils.plot_model(model, "E:/Computer Vision and Machine Learning Projects/BuildingCNN/model_plot.png", show_shapes=True, show_layer_names=True )

#saving model
model.save("E:/Computer Vision and Machine Learning Projects/BuildingCNN/monkey_breed_classifier_5ep.h5")

#Saving History of Model
pickle_out = open("history.pickle", "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

#Loading history of model
pickle_in = open("history.pickle", "rb")
saved_history = pickle.load(pickle_in)
print(saved_history)

#plotting model
#Loss Plot
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
ep = range(1, len(loss_values)+1)
line1 = plt.plot(ep, val_loss_values, label = "Validation Loss")
line2 = plt.plot(ep, loss_values, label = "Training Loss")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

#Accuracy Plot
loss_values = history_dict["accuracy"]
val_loss_values = history_dict["val_accuracy"]
ep = range(1, len(loss_values)+1)
line1 = plt.plot(ep, val_loss_values, label = "Validation Accuracy")
line2 = plt.plot(ep, loss_values, label = "Training Accuracy")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()