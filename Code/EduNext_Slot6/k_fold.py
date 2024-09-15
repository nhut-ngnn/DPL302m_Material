from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten, LSTM, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import os
from Preprocessing.data_augmentation import *

WIDTH = 128
HEIGHT = 128
IMG_SIZE = (WIDTH, HEIGHT)
BATCH = 32
K = 5  # Number of folds

def load_data(train_generator):
    X, y = [], []
    for i in range(len(train_generator)):
        X_batch, y_batch = train_generator[i]
        X.append(X_batch)
        y.append(y_batch)
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y

def create_model_with_lstm(input_shape):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fourth Convolutional Block
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Reshape and LSTM Layers
    model.add(Flatten())
    model.add(Reshape((-1, 256)))  # Reshape to (timesteps, features)
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))

    # Fully Connected Layers
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def train_k_fold(train_generator, k=5):
    X, y = load_data(train_generator)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    input_shape = (WIDTH, HEIGHT, 3)
    fold_no = 1

    for train_index, val_index in kf.split(X):
        print(f'Training fold {fold_no}...')
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model = create_model_with_lstm(input_shape)
        
        models_path = 'C:/Users/admin/Documents/DPL302m/DPL302m_Material/Code/EduNext_Slot6/Models/K_fold/'
        os.makedirs(models_path, exist_ok=True)
        
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                                    patience=4, 
                                                    verbose=1, 
                                                    factor=0.75, 
                                                    min_lr=0.00001)

        early_stopping = EarlyStopping(patience=10)

        checkpoint_acc = ModelCheckpoint(filepath=models_path + f'model_best_acc_fold_{fold_no}.keras',
                                         monitor="val_accuracy",
                                         save_best_only=True, save_freq='epoch')

        csv_logger = CSVLogger(models_path + f'training_log_fold_{fold_no}.csv', append=True)

        callbacks = [learning_rate_reduction, early_stopping, checkpoint_acc, csv_logger]

        EPOCHS = 10
        BATCH_SIZE = 32

        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        # Save the model
        model.save(models_path + f'model_fold_{fold_no}.h5')
        
        fold_no += 1

# Assuming train_generator is already defined
train_k_fold(train_generator, k=K)