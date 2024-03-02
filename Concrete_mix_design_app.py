import numpy as np
import joblib as jb
import streamlit as st

def scaler(input_features):
    linear_scaler = jb.load("Scaling_function.pkl")
    scaled_input_features = linear_scaler.transform(input_features)
    return scaled_input_features

def prediction(scaled_input_features):
    model = jb.load('RandomForest_model_97%_accuracy')
    predicted_values = model.predict(scaled_input_features)
    return predicted_values

def main():
    input_names = ['Fine Aggregate Water Absoption', 'Fine Aggregate unit weight', 'Coarse Aggregate Water Absorption',
                   'Coarse Aggregate unit weight ', 'Required Slump', 'Required 28 days compressive strength',
                   'Coarse Aggregate size in mm']
    
    fine_water = st.text_input("Fine Aggregate Water Absoption")
    fine_unit = st.text_input("Fine Aggregate Unit Weight kg/m^3")
    coarse_water = st.text_input("Coarse Aggregate Water Absorption")
    coarse_unit = st.text_input("Coarse Aggregate unit weight kg/m^3")
    slump = st.text_input("Required Slump")
    strength= st.text_input("Required 28 days compressive strength in psi")
    size = st.text_input("Coarse Aggregate size in mm")

    # Convert input fields to float, handling empty strings
    input_values = [fine_water, fine_unit, coarse_water, coarse_unit, slump, strength, size]
    input_values = [float(value) if value else np.nan for value in input_values]
    input_features = np.array([input_values])

    # Remove rows with NaN values
    input_features = input_features[~np.isnan(input_features).any(axis=1)]

    # Perform prediction only if input features are not empty
    if len(input_features) > 0:
        scaled_input_features = scaler(input_features)
        predicted_values = prediction(scaled_input_features)

        button = st.button('Predict')
        if button:
            if predicted_values.size > 0:
                st.write(f'### {predicted_values[0][0] / predicted_values[0][0]:.2f} ,    '
                        f' {predicted_values[0][1] / predicted_values[0][0]:.2f} ,     '
                        f' {predicted_values[0][2] / predicted_values[0][0]:.2f},   @ w/c  '
                        f' {predicted_values[0][3] / predicted_values[0][0]:.2f}\n'
                        f'### 7 days predicted strength = {predicted_values[0][4]:.2f} psi')
            else:
                st.warning("No prediction available. Please fill in all input fields.")
    
if __name__ == "__main__":
    main()
