from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from data_augmentation import *

import time

def create_resnet50_model(input_shape=(WIDTH, HEIGHT, 3)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

# Example usage
model = create_resnet50_model()

models_path = 'C:/Users/admin/Documents/DPL302m/DPL302m_Material/Code/EduNext_Slot6/Models/ResNet50/'
# os.makedirs(models_path)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.75, 
                                            min_lr=0.00001)

early_stopping = EarlyStopping(patience=10)

checkpoint_acc = ModelCheckpoint(filepath=models_path + 'model_best_acc.keras',
    monitor="val_accuracy",
    save_best_only=True, save_freq='epoch')

callbacks = [learning_rate_reduction, early_stopping, checkpoint_acc]

EPOCHS = 10
beg = int(time.time())

history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH,
        callbacks=callbacks)

end = int(time.time())
t = end - beg
hrs = t // 3600
mins = (t - 3600 * hrs) // 60
secs = t % 60
print("training took {} hrs -- {} mins -- {} secs".format(hrs, mins, secs))
model.save_weights(models_path + 'ResNet50.weights.h5')
model.save(models_path + 'ResNet50.h5')