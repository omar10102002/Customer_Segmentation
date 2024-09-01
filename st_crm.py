import streamlit as st
import pickle
import numpy as np

# Load the model
kmeans = pickle.load(open('E:\Epsilon Ai\Omar Hamad\Final Poject\crm_prediction.sav', 'rb'))

st.title("Prediction of Customer Segmentation")

recency = st.number_input('Enter Recency:How recent the customers last purchase was. The closer the purchase, the higher the score.')
frequency = st.number_input('Enter Frequency:How often the customer makes purchases. More frequent purchases result in a higher score.')
monetary = st.number_input('Enter Monetary:The total amount of money the customer has spent. Higher spending results in a higher score.')
# days_gap = st.number_input('Enter Days Gap:')

# When the button is clicked
if st.button('Predict Cluster'):
    # Prepare the input data for the KMeans model
    input_data = np.array([[recency, frequency, monetary]])
    
    # Make prediction using the model
    cluster = kmeans.predict(input_data)
    
    # Display the result
    st.write(f"The predicted cluster is: {cluster[0]}")
