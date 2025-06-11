from keras import Sequential, Input, layers
import numpy as np
from pocketcoach.params import *
from pathlib import Path
from pocketcoach.dl_logic.data import get_data
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def train_base_model(
    X_train,
    y_train,
    X_val,
    y_val,
    vocab_size
):
    """
    Trains a base model on the training set
    """

    print("Creating model architecture")
    embedding_size = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, mask_zero=True))
    model.add(layers.Conv1D(16, kernel_size=3))
    model.add(layers.Flatten())
    model.add(layers.Dense(5,))
    model.add(layers.Dense(6, activation='softmax'))


    print("Compiling model")
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print("Fitting model")
    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[es]
    )

    return model
