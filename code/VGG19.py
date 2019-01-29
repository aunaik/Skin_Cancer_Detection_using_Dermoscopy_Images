# Referred the following links for understanding implementation of Transfer Learning:
# https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from keras import backend as K
import keras
import cv2
import sys
import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import model_from_json
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

directory = "/N/u/aunaik/BigRed2/"

X = []
y = []

# for first class
count = 0
for filename in os.listdir(directory + "benign_224"):
    if count > 1000:                   # To limit the number of input images for training
        break
    count += 1
    
    img = cv2.imread(directory + "benign_224/" + filename)
    X.append(img)
    y.append(0)

# for second class    
count = 0
for filename in os.listdir(directory + "malignant_224"):
    if count > 1000:                   # To limit the number of input images for training
        break
    count += 1

    img = cv2.imread(directory + "malignant_224/" + filename)
    X.append(img)
    y.append(1)

p = np.random.permutation(len(X))
p_train = p[0:int(0.95*(len(X)))]
p_test = p[int(0.95*len(X)):]

X = np.array(X)  
y = np.array(y)

X_train = X[p_train]
y_train = y[p_train]

X_test = X[p_test]
y_test = y[p_test]

img_width = 224
img_height = 224
batch_size = 64
epochs = 100

model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

model.summary()

for layer in model.layers[:5]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dropout(0.7)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.7)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)

# creating the final model 
model_final = Model(inputs = model.input, outputs = predictions)

model_final.summary()

# compile the model 
model_final.compile(loss="binary_crossentropy", optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=["accuracy"])

# Fitting the train data
history = model_final.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

# Evaluating accuracy
model_final.evaluate(x=X_train, y=y_train, batch_size=None, verbose=1, sample_weight=None)


# Prediction class labels for each train image
y_pred = model_final.predict(X_train)
y_pred = np.rint(y_pred)
y_pred = y_pred.astype(int)
print (y_pred)

y_pred = y_pred.flatten()
y_pred = y_pred.astype(int)
correct = 0

print (y_train)
print (y_pred)

for i in range(len(y_train)):
  if y_train[i] == y_pred[i]:
    correct += 1

accuracy = 100.0*correct/len(y_train)

print (accuracy)

# Prediction class labels for each test image
y_pred = model_final.predict(X_test)
y_pred = np.rint(y_pred)
y_pred = y_pred.astype(int)
print (y_pred)

y_pred = y_pred.flatten()
y_pred = y_pred.astype(int)
correct = 0

print (y_test)
print (y_pred)

bad_test = []

for i in range(len(y_test)):
  if y_test[i] == y_pred[i]:
    correct += 1
  elif y_test[i]:
    im = Image.fromarray(X_test[i])
    im.save("/N/u/ninjoshi/BigRed2/wrong_class/wrong_" + str(i) + ".png")

 
accuracy = 100.0*correct/len(y_test)

print (accuracy)
print ("Accuracy from sklearn: " + str(metrics.accuracy_score(y_test, y_pred)))
print ("Recall from sklearn: " + str(metrics.recall_score(y_test, y_pred)))

# Plotting Accuracy graph
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train set', 'test set'], loc='best')
plt.title('Train vs Validation Accuracy')


# Plotting loss graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train set', 'test set'], loc='best')
plt.tight_layout()
plt.savefig('VGG-19.png')

# Saving the trained model
VGG19_model_json = model.to_json()
with open("VGG19_model.json", "w") as json_file:
    json_file.write(VGG19_model_json)

# Saving the trained weights
model.save_weights("VGG19_model.h5")


# Loading the model with pre-trained weights
# json_file = open('VGG19_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("VGG19_model.h5")
