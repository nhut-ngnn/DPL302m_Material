# -*- coding: utf-8 -*-
"""KerasTuner_hyperparameters_MNIST.ipynb


"""

#installing keras-tuner
!pip install keras-tuner

#importing necessary liabraries
import tensorflow as tf
import keras_tuner
import matplotlib.pyplot as plt
import numpy as np

#loading dataset and spliting it in test and train dataset
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.fashion_mnist.load_data()

#check x_train shape
x_train.shape

#setting the y_train data
set(y_train)

#checking a random element 7 and visualizing it
plt.imshow(x_train[7],cmap='binary')
plt.xlabel(y_train[7])
plt.show()

#creating model for the tuner

def create_model(hp):
  num_hidden_layers=1 #hidden layer
  num_units=8
  dropout_rate=0.1 #dropout rate
  learning_rate=0.01 #learning_Rate

  if hp:  #creating a hyperparmeter with choices
    num_hidden_layers=hp.Choice('num_hidden_layers',values=[1,2,3])
    num_units=hp.Choice('num_units',values=[8,16,32])
    dropout_rate=hp.Float('dropout_rate',min_value=0.1,max_value=0.5)
    learning_rate=hp.Float('learning_rate',min_value=0.0001,max_value=0.01)

  model=tf.keras.models.Sequential() #creating a sequential model

  model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #flatten the layer
  model.add(tf.keras.layers.Lambda(lambda x: x/255.))

  for _ in range(0, num_hidden_layers):
    model.add(tf.keras.layers.Dense(num_units,activation='relu')) #relu activation
    model.add(tf.keras.layers.Dropout(dropout_rate))

  model.add(tf.keras.layers.Dense(10,activation='softmax')) #softmax activation

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      metrics=['accuracy']
  )
  return model

#creating model summary
create_model(None).summary()

#defining a class with custom tuner using bayesian optimzation
class CustomTuner(keras_tuner.tuners.BayesianOptimization):
  def run_trial(self,trial, *args, **kwargs):
    kwargs['batch_size']=trial.hyperparameters.Int('batch_size',32,128,step=32) #giving batch size
    super(CustomTuner,self).run_trial(trial,*args,**kwargs)

#running a custom tuner
tuner=CustomTuner(
    create_model,
    objective='val_accuracy', #validation accuracy
    max_trials=20,           #defining max number of trials
    directory='logs',
    project_name='fashion_mnist',
    overwrite=True
)

tuner.search_space_summary()

tuner.search(
    x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=5,verbose=False
)

tuner.results_summary(1)

model=tuner.get_best_models(num_models=1)[0]
model.summary()

fit = model.fit(
    x_train,y_train,
    validation_data=(x_test,y_test),
    epochs=20,batch_size=128,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=3)]
)