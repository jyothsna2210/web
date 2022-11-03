# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:51:53 2022

@author: Jyothsna
"""

import numpy as np
import pickle
loaded_model=pickle.load(open("C:/Users/Jyothsna/OneDrive/Desktop/test/house_model.sav",'rb'))
input_data=(0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.0900,1.0,296.0,15.3,393.90,4.98)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)




















