import numpy as np
import joblib as jb
import streamlit as st
from datetime import datetime
import time
from streamlit_option_menu import option_menu

#This will scale complete data including input data and predicted data.

def scaler_day28(input_data):
    scaler_28days = jb.load("scaler_for_28days_model.pkl")
    scaled_data = scaler_28days.transform(input_data)
    return scaled_data

def day28(scaled_data):
    strength_model = jb.load("28Days_strength_model_98%A")
    strength = strength_model.predict(scaled_data)
    return strength




# This is the function which load StandardScaler

def scaler(input_features):
    linear_scaler = jb.load("Scaling_function.pkl")
    scaled_input_features = linear_scaler.transform(input_features)
    return scaled_input_features

# This is function which load model and predict the output

def prediction(scaled_input_features):
    model = jb.load('RandomForest_model_97%_accuracy')
    predicted_values = model.predict(scaled_input_features)
    return predicted_values

#This is main page of the streamlit app

def main():

    ### This shows the greeting on the side bar depending upon the time


    # greeting_message = ""
    # current_hour = datetime.now().hour
    # if 6 <= current_hour < 12:
    #     greeting_message = "Good morning"
    # elif 12 <= current_hour < 18:
    #     greeting_message = "Good afternoon"
    # else:
    #     greeting_message = "Good evening"

    # st.sidebar.write(f"# {greeting_message}!")


    # ## This is introduce my self on the side bar
    # def animated_into(intro_text):
    
    #     text_slot = st.sidebar.empty()
    #     typed_text = ""
    #     for char in intro_text:
    #         typed_text += char
    #         text_slot.markdown(f"<pre>{typed_text}</pre>", unsafe_allow_html=True)
    #         time.sleep(0.01)
    
    # introduction = "Hello! Welcome to my streamlit app, I am Mustansar Hussain currently doing my bachlers in Civil Engineering"
    # st.markdown(
    # """
    # <style>
    # .sidebar .sidebar-content {
    #     border-bottom: 1px solid #ccc;
    # }
    # </style>
    # """,
    # unsafe_allow_html=True)

    # st.sidebar.title("Indroduction")
    # animated_into(introduction)






    ###This is showing different tabs in the side bar
    with st.sidebar:
        option_click = option_menu(
            menu_title="Main Menu",
            options=["Main Page", "Projects", "Contact"],
        )
    



    input_names = ['Fine Aggregate Water Absoption', 'Fine Aggregate unit weight', 'Coarse Aggregate Water Absorption',
                   'Coarse Aggregate unit weight ', 'Required Slump', 'Required 28 days compressive strength',
                   'Coarse Aggregate size in mm']
    
    st.title("Normal Concrete Mix Design App")
    st.write("#### Introduction :")
    st.write(
            " This is the app for the prediction of the concrete mix design data which include the prediction of"
            " **Cement, Fine Aggregate, Coarse Aggregate, and Water**. In simple word it mean that this app is"
            " resposible for calculation ratio. Here you have to provide certain input data which include; "
            " **__Fine Aggregate water abosorption, Fine Aggregate unit weight, Coarse Aggregate water absorption, "
            " Coarse Aggregate unit weight, size of coarse aggregate, required Slump, and 28 Days compressive strength__**")
    st.write("#### Accuracy: ")
    st.write(" This model is developed using scikit-learn library on the **Random Forest Machine Learning Model**"
             " This model has **accuracy arround 97%:**")
    
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
                st.write(f'***{predicted_values[0][0] / predicted_values[0][0]:.2f}*** ,    '
                        f' ***{predicted_values[0][1] / predicted_values[0][0]:.2f}*** ,     '
                        f' ***{predicted_values[0][2] / predicted_values[0][0]:.2f}***,   ***@ w/c***  '
                        f' ***{predicted_values[0][3] / predicted_values[0][0]:.2f}***\n'
                        f' 7 days predicted strength = ***{predicted_values[0][4]:.2f} psi***')
                        # f'#### 28 days predicted strength from predicted data is {strength28days} psi')
                all_features = np.array([[fine_water, fine_unit, coarse_water, coarse_unit, predicted_values[0][0],predicted_values[0][1],
                                         predicted_values[0][2],predicted_values[0][3], slump, size
                                         ]])
                scaled_all_features = scaler_day28(all_features)
                strength28days = day28(scaled_all_features)
                st.write(f' 28 days predicted strength from predicted data is = ***{strength28days} psi***')
                
            else:
                st.warning("No prediction available. Please fill in all input fields.")
    
if __name__ == "__main__":
    main()
