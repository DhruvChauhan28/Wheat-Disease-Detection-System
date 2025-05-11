import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Load the model
model = load_model('wheat_disease_model')
model.summary()

# Load validation set to get class names
from tensorflow.keras.preprocessing.image import ImageDataGenerator

valid_datagen = ImageDataGenerator(rescale=1./255)
validating_set = valid_datagen.flow_from_directory(
    'data/valid', class_mode='categorical', target_size=(128, 128), batch_size=32)

class_names = list(validating_set.class_indices.keys())
print("Class Names:", class_names)

# Load and preprocess image
image_path = r"D:\drive d\study material\3rd year study material\sem 6\design engineering\data\test\healthy_test\healty_test_49.png"

if os.path.exists(image_path):
    print("File exists!")
else:
    print("Error: File not found. Check the path!")
    exit()

# Read image using OpenCV
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Display the image
plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

# Preprocess image
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128,128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = input_arr / 255.0  # Normalize the image
input_arr = np.expand_dims(input_arr, axis=0)  # Ensure correct shape

# Predict
predictions = model.predict(input_arr)
result_index = np.argmax(predictions)
model_prediction = class_names[result_index]

# Display the prediction result
plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()

# Show top 3 predictions with probabilities
top_3_indices = np.argsort(predictions[0])[-3:][::-1]
print("\nTop Predictions:")
for i in top_3_indices:
    print(f"{class_names[i]}: {predictions[0][i]*100:.2f}%")
