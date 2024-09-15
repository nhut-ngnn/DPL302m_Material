from tensorflow.keras.models import load_model
from data_augmentation import *

model_path = "C:/Users/admin/Documents/DPL302m/DPL302m_Material/Code/EduNext_Slot6/Models/ResNet50/ResNet50.h5"
model = load_model(model_path)
scores = model.evaluate(test_generator)
