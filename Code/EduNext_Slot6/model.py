from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from Preprocessing.data_augmentation import *
from utils.utils import model_dir
import time

def create_model(input_shape=(WIDTH, HEIGHT, 3)):
    inputs = Input(shape=input_shape)

    # First branch
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(0.25)(x1)

    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(0.25)(x1)

    x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(0.25)(x1)

    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(0.25)(x1)

    x1 = Flatten()(x1)

    # Second branch
    x2 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.25)(x2)

    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.25)(x2)

    x2 = Conv2D(128, (5, 5), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.25)(x2)

    x2 = Conv2D(256, (5, 5), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Dropout(0.25)(x2)

    x2 = Flatten()(x2)

    # Feature fusion
    x = Concatenate()([x1, x2])

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

# Example usage
model = create_model()
models_path = model_dir() + '/model/'
# os.makedirs(models_path)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience= 4, 
                                            verbose= 1, 
                                            factor= 0.75, 
                                            min_lr= 0.00001)

early_stopping = EarlyStopping(patience = 15)

checkpoint_acc = ModelCheckpoint(filepath = models_path + 'model_best_acc.keras',
    monitor = "val_accuracy",
    save_best_only = True, save_freq= 'epoch' )

filename = models_path + 'log.csv'
history_logger = CSVLogger(filename, separator=",", append=True)

callbacks = [learning_rate_reduction, early_stopping, checkpoint_acc, history_logger]

EPOCHS = 60
beg = int(time.time())
BATCH_SIZE = 32

# Calculate steps_per_epoch and validation_steps
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

history = model.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        epochs = EPOCHS,
        validation_data = validation_generator,
        validation_steps = validation_steps,
        callbacks = callbacks)

end = int(time.time())
t = end - beg
hrs = t // 3600
mins = (t - 3600 * hrs) // 60
secs = t % 60
print("training took {} hrs -- {} mins -- {} secs".format(hrs, mins, secs))
model.save_weights(models_path + 'modelCNN.weights.h5')
model.save(models_path + 'modelCNN.h5')