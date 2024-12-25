import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Estimated Salary Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
is_exited = st.selectbox('Is Exited', [0, 1])

input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography], 
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [is_exited]
})

input_df

geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()

geo_encoded

geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.drop('Geography', axis = 1), geo_encoded_df], axis = 1)

input_data_scaled = scaler.transform(input_df)

prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: {prediction_salary: .2f}')