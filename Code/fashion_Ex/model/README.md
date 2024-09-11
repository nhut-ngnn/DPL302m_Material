# Saved Models Folder

This folder contains the saved models for the Fashion-MNIST project. These models have been trained and saved for later use, evaluation, or deployment.

## Contents

- **model_1.h5**: The first trained model saved in HDF5 format.
- **model_2.h5**: The second trained model saved in HDF5 format.
- **model_3.h5**: The third trained model saved in HDF5 format.
- **README.md**: This file, providing an overview of the contents and purpose of the `saved_models` folder.

## Usage

1. **Loading Models**: You can load these models using the `load_model` function from Keras.
    ```python
    from keras.models import load_model

    model = load_model('path/to/saved_models/model_1.h5')
    ```

2. **Evaluating Models**: Evaluate the loaded models on your test dataset.
    ```python
    model.evaluate(X_test, y_test)
    ```

3. **Making Predictions**: Use the loaded models to make predictions on new data.
    ```python
    predictions = model.predict(new_data)
    ```

## Notes

- Ensure that the paths to the saved models are correct when loading them.
- Modify the scripts as needed for your specific use case.
