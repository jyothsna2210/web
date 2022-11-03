# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 12:01:38 2022

@author: Jyothsna
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
loaded_model=pickle.load(open("C:/Users/Jyothsna/OneDrive/Desktop/test/house_model.sav",'rb'))
def houseprice(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    return prediction
def main():
    st.title("House price prediction")
    html_temp="""
    <div style="background-color:yellow;padding:13px">
    <h1 style="color:black;text-align:center;">Streamlit House price prediction </h1>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    CRIM=st.text_input('crim')
    ZN=st.text_input('zn')
    INDUS=st.text_input('indus')
    CHAS=st.text_input('chas')
    NOX=st.text_input('nox')
    RM=st.text_input('rm')
    AGE=st.text_input('age')
    DIS=st.text_input('dis')
    RAD=st.text_input('rad')
    TAX=st.text_input('tax')
    PTRATIO=st.text_input('ptratio')
    B=st.text_input('b')
    LSTAT=st.text_input('lstat')
    result=""
    if st.button('predict'):
        result = houseprice([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,
           PTRATIO, B, LSTAT]])
    st.success(result)
if __name__=='__main__':
    main()
    












 