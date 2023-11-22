#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('preprocessed_data.csv')

train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Prepare Data
X = train_df['Close'].values
y = train_df['Close'].shift(-1).fillna(0).values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Reshape Data
s_len = 24
X_train = np.reshape(X_train[:len(X_train)//s_len * s_len],
                     (len(X_train)//s_len, s_len, 1))
y_train = np.reshape(y_train[:len(y_train)//s_len * s_len],
                     (len(y_train)//s_len, s_len, 1))

X_test = np.reshape(X_test[:len(X_test)//s_len * s_len],
                    (len(X_test)//s_len, s_len, 1))
y_test = np.reshape(y_test[:len(y_test)//s_len * s_len],
                    (len(y_test)//s_len, s_len, 1))

# Build RNN
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(24, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    tf.keras.layers.LSTM(24, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train
model.fit(X_train,
          y_train,
          epochs=20,
          batch_size=32,
          validation_data=(X_test, y_test))

# Evaluate
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Data: {loss}')
