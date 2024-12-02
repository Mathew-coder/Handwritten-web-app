import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers,Sequential
import pickle
import time
from tensorflow.keras.models import load_model

start_time= time.time()

data=keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()

#Normalize pixel values to be between 0 and 1

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test, y_train, y_test= train_test_split(x_train, y_train, test_size=0.1, random_state=42)

#CNN Model

cnn_model = keras.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

cnn_accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)[1]
print(f"CNN Accuracy: {cnn_accuracy}")

#### SAVE THE TRAINED MODEL 
# pickle_out= open("cnn_model.p", "wb")
# pickle.dump(cnn_model,pickle_out)
# pickle_out.close()

cnn_model.save('cnn_model.h5')
           
end_time=time.time()

tot_time=end_time-start_time

print(f"Total time taken {tot_time}")
