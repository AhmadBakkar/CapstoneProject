import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump

filename = "trained_model_awarded.joblib"
#filename1 = "trained_model_2.sav"

#with open(filename, 'rb') as f:
#    model = pickle.load(f)
model = joblib.load("trained_model_awarded.joblib")

#with open(filename1,'rb') as f1:
#    model1 = pickle.load(f1)


# Define your preprocessor pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=np.number, dtype_exclude='timedelta64')),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])

# Function to preprocess input data
def preprocess_input_data(input_data):
    processed_data = preprocessor.transform(input_data)
    return processed_data

# Function to make predictions
def predict(input_data):
    # Preprocess the input data
    processed_data = preprocess_input_data(input_data)
    
    # Make predictions using the loaded model
    predictions = model.predict(processed_data)
    return predictions

# Streamlit app
def main_view():
    st.title('Model 1')
    st.write('Please Insert your file to check whether to project will get awarded or not')

    # Sidebar for user input
    st.sidebar.header('User Input')
    # Add input widgets for user input data
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write('Input Data:')
        st.write(input_data)

        # Make predictions
        predictions = model.predict(input_data)

        # Map numerical predictions to labels
        prediction_labels = ['Awarded' if pred == 0 else 'Lost' for pred in predictions]
        # Display predictions
        st.header('Predictions:')
        st.write(prediction_labels)

# Function to display the model information view
def model_info_view():
    st.title('Model 2')
    st.write('Please Insert your file to check the Project Budget Class')
    # Add code to display model information here
    # Sidebar for user input
    st.sidebar.header('User Input')
    # Add input widgets for user input data
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.write('Input Data:')
        st.write(input_data)

        # Make predictions
        predictions = model1.predict(input_data)

        # Map numerical predictions to labels
        label_mapping = {
            0: 'Low',
            1: 'Medium',
            2: 'High',
            3: 'Very High',
            4: 'MEGA'
        }
        prediction_labels1 = [label_mapping[pred] for pred in predictions]
        # Display predictions
        st.header('Predictions:')
        st.write(prediction_labels1)


# Main function to run the app
def main():
    # Add a selectbox to allow users to select the view
    view_options = ['Model 1', 'Model 2']
    view = st.sidebar.selectbox('View', view_options)

    # Display the selected view
    if view == 'Model 1':
        main_view()
    elif view == 'Model 2':
        model_info_view()

if __name__ == '__main__':
    main()
