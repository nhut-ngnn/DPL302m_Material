# Import the ImageDataGenerator class
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import *

WIDTH = 128
HEIGHT = 128
IMG_SIZE = (WIDTH , HEIGHT)
BATCH = 32

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(directory= train_dir(), 
                                                    target_size = IMG_SIZE,
                                                    classes=['cat' , 'dog'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    #save_to_dir=aug_data_path,
                                                    #save_prefix='aug_',
                                                    #save_format="jpg",
                                                    seed = 1
                                                    
                                                    )
print(train_generator.class_indices)

validation_generator = validation_datagen.flow_from_directory(directory= validation_dir(), 
                                                    target_size = IMG_SIZE,
                                                    classes=['cat' , 'dog'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    seed = 1 )
print(validation_generator.class_indices)

test_generator = test_datagen.flow_from_directory(directory= test_dir(), 
                                                    target_size = IMG_SIZE,
                                                    classes=['cat' , 'dog'],
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    seed = 1 )
print(test_generator.class_indices)