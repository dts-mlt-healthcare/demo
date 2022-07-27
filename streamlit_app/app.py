import streamlit as st
import tensorflow as tf 
import pandas as pd
import joblib
from predict import pred_disease

def load_data():
    desc_data = pd.read_csv('data_healthcare/symptom_Description.csv')
    prec_data = pd.read_csv('data_healthcare/symptom_precaution.csv')
    model = tf.keras.models.load_model('model/Classification_Disease.h5')
    enc = joblib.load('model/label_ohe.pkl')
    mlb = joblib.load('model/symps_ohe.pkl')
    symps_data = mlb.classes_.tolist()
    return desc_data, prec_data, model, enc, symps_data

desc_data, prec_data, model, enc, symps_data = load_data()

st.title('Disease Classification \n **using Neural Network**')
st.code("Final Project DTS PROA - team(Healthcare_01)")
st.subheader("what symptoms do you have?")

# Input symptoms with multiselectet
symps = st.multiselect('',symps_data, help="Choose symptoms")

if st.button("Disease Classification"):
    result, proba, desc, prec = pred_disease(symps, 
                                        desc_data, 
                                        prec_data, model, 
                                        enc, symps_data)
    if proba < 0.75: 
        st.warning(f"you got {'{:.2%}'.format(proba)} {result}, please add another symptom for better result")
    st.header(result)
    st.subheader("{:.2%}".format(proba))
    with st.expander("See Disease explanation"):
        st.write(desc)
    with st.expander("See Precaution"):
        st.write(f"Precaution 1 : {prec[0]}")
        st.write(f"Precaution 2 : {prec[1]}")
        st.write(f"Precaution 3 : {prec[2]}")
        st.write(f"Precaution 4 : {prec[3]}")


