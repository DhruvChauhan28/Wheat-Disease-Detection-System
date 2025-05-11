# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:11:31 2025

@author: chauh
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense , Flatten
import tensorflow.keras.layers 
# making the whole model.
classifier = Sequential()
classifier.add(Conv2D(32 , (3,3) , input_shape = (128,128,3) , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2) , strides = (2,2)))

classifier.add(Conv2D(64 , (3,3) , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

classifier.add(Conv2D(128, (3,3) , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2) ,strides = (2,2)))

classifier.add(Conv2D(256, (3,3) , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

classifier.add(tensorflow.keras.layers.Dropout(0.25))
classifier.add(Flatten())

# adding the full connected layer.
classifier.add(Dense(units = 1200 , activation= 'relu'))
classifier.add(tensorflow.keras.layers.Dropout(0.4))
classifier.add(Dense(units = 15 , activation= 'softmax'))

# compiling the cnn
classifier.compile(optimizer = 'adam' , loss ='categorical_crossentropy', metrics = ['accuracy'])

# image augmentation and traing the model.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                    rescale=1.0/255,  # Normalizing pixel values
                    shear_range=0.2,  # Shear transformation
                    zoom_range=0.2,  # Random zoom
                    horizontal_flip=True,  # Horizontal flipping
                    )

valid_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
                        'data/train',
                        class_mode = 'categorical',
                        target_size = (128,128),
                        batch_size = 32)

validating_set = valid_datagen.flow_from_directory(
                        'data/valid',
                        class_mode = 'categorical',
                        target_size = (128,128),
                        batch_size = 32)


with tf.device('/GPU:0'):
    history = classifier.fit(training_set , 
               steps_per_epoch = len(training_set),
               validation_data = validating_set,
               epochs = 50,
               validation_steps = len(validating_set),
               )


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

print("Accuracy : " , history.history['accuracy'][-1])
print("Val_accuracy : " , history.history['val_accuracy'][-1])
print("loss : " , history.history['loss'][-1])
print("Val_loss : " , history.history['val_loss'][-1])


#training_set_accuracy
train_loss, train_acc = classifier.evaluate(training_set)
print('Training accuracy:', train_acc)
#Validation set Accuracy
val_loss, val_acc = classifier.evaluate(validating_set)
print('Validation accuracy:', val_acc)

#visulization of model using graph.
epochs = [i for i in range(1,51)]
plt.plot(epochs,history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Visualization of Accuracy Result')
plt.legend()
plt.show()

#writing training data into json.
#Recording History in json
import json
with open('training_hist.json','w') as f:
  json.dump(history.history,f)
  
print(history.history.keys())

validating_set = valid_datagen.flow_from_directory(
    'data/valid',
    class_mode='categorical',
    target_size=(128, 128),
    batch_size=32,
    shuffle=False   # <-- Important!
)

#some other metrices for model evaluation.
import seaborn as sns
class_name = list(validating_set.class_indices.keys())
# RESET generator before predictions
validating_set.reset()

# Get predictions
y_pred = classifier.predict(validating_set, steps=len(validating_set), verbose=1)
predicted_categories = tf.argmax(y_pred, axis=1).numpy()

# RESET again before extracting labels
validating_set.reset()

# Extract true labels properly
true_labels = []
for i in range(len(validating_set)):
    _, labels = validating_set[i]
    true_labels.extend(labels)

Y_true = tf.argmax(tf.convert_to_tensor(true_labels), axis=1).numpy()
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Accuracy:", accuracy_score(Y_true, predicted_categories))
print(classification_report(Y_true, predicted_categories, target_names=class_name))

# Optional: Confusion matrix
cm = confusion_matrix(Y_true, predicted_categories)


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_name, yticklabels=class_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


#plotting confusion matrix visualization.
plt.figure(figsize=(40, 40))
sns.heatmap(cm,annot=True,annot_kws={"size": 10})
plt.xlabel('Predicted Class',fontsize = 20)
plt.ylabel('Actual Class',fontsize = 20)
plt.title('Wheat Disease Prediction Confusion Matrix',fontsize = 25)
plt.show()

classifier.save('wheat_disease_final_model')


