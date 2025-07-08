import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

import sklearn.model_selection as sk
from sklearn.model_selection import train_test_split


from cnn_model import build_model, plot_history


from load_GTZAN_data import load_all_data, make_gatzan_img_data, get_my_gatzan_training_data 


# gets training data
X, y = get_my_gatzan_training_data()
# print(f"size of X and y data is {X.shape} and {y.shape}")

# splits training data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
input_shape = (X_train.shape[1], X_train.shape[2], 3) # skips [0] because that is 999(number of examples)
print(f"inptut shape is {input_shape}")

# build model 
model = build_model(input_shape)

# [overfit]: This will stop training if val_loss doesn't improve for 3 epochs in a row
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# compile model
model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy'],
    callback=[early_stopping]
)
    

model.summary()

# train model

history = model.fit( 
        X_train, y_train,
        epochs=16,          # How many times the model will see the entire training dataset.
        validation_data=(X_val, y_val),
        verbose=1               # Shows a progress bar and training metrics per epoch.
        callbacks=[early_stopping] # Add the callback here
)

plot_history(history)

model.save("my_gatzan_cnn_model.h5")
