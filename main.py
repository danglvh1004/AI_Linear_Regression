# !/usr/bin/env python
# coding: utf-8

# # CODED BY HAI DANG LE VU
# # INPUT
# + ENGINESIZE
# + CYLINDERS
# + FUELCONSUMPTION_CITY
# + FUELCONSUMPTION_HWY
# + FUELCONSUMPTION_COMB
# + FUELCONSUMPTION_COMB_MPG

# # OUTPUT: CO2EMISSIONS

# In[34]:


import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import r2_score
import numpy as np
import statsmodels.api as sm

# In[35]:


data = pd.read_csv("FuelConsumptionCo2.csv")
data.head()

# In[36]:


# Input data to training and test
train = data[:(int((len(data) * 0.8)))]
test = data[(int((len(data) * 0.8))):]

# In[37]:


# Input
X = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY',
           'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG']]
# Output
Y = train["CO2EMISSIONS"]

# In[38]:


# with sklearn
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# In[39]:


test_x = np.array(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                        'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG']])
test_y = np.array(test["CO2EMISSIONS"])
Y_pred = regr.predict(test_x)
R = r2_score(test_y, Y_pred)
print("R² :", R)

# In[40]:


# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

# In[41]:


# prediction with sklearn
ENGINESIZE = 2
CYLINDERS = 4
FUELCONSUMPTION_CITY = 10.2
FUELCONSUMPTION_HWY = 7.5
FUELCONSUMPTION_COMB = 9
FUELCONSUMPTION_COMB_MPG = 31
print('CO2EMISSIONS: \n', regr.predict([[ENGINESIZE, CYLINDERS, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY,
                                         FUELCONSUMPTION_COMB, FUELCONSUMPTION_COMB_MPG]]))

# In[32]:


# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=1000, height=800)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
canvas1.create_window(260, 230, window=label_Intercept)

# with sklearn
Coefficients_result = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
canvas1.create_window(260, 270, window=label_Coefficients)

# ENGINESIZE label and input box
label1 = tk.Label(root, text=' ENGINESIZE: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# CYLINDERS label and input box
label2 = tk.Label(root, text=' CYLINDERS: ')
canvas1.create_window(100, 120, window=label2)

entry2 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

# FUELCONSUMPTION_CITY label and input box
label3 = tk.Label(root, text=' FUELCONSUMPTION_CITY: ')
canvas1.create_window(100, 140, window=label3)

entry3 = tk.Entry(root)  # create 3rd entry box
canvas1.create_window(270, 140, window=entry3)

# FUELCONSUMPTION_HWY label and input box
label4 = tk.Label(root, text=' FUELCONSUMPTION_HWY: ')
canvas1.create_window(100, 160, window=label4)

entry4 = tk.Entry(root)  # create 4th entry box
canvas1.create_window(270, 160, window=entry4)

# FUELCONSUMPTION_COMB label and input box
label5 = tk.Label(root, text=' FUELCONSUMPTION_COMB: ')
canvas1.create_window(100, 180, window=label5)

entry5 = tk.Entry(root)  # create 5th entry box
canvas1.create_window(270, 180, window=entry5)

# FUELCONSUMPTION_COMB_MPG label and input box
label6 = tk.Label(root, text=' FUELCONSUMPTION_COMB_MPG: ')
canvas1.create_window(100, 200, window=label6)

entry6 = tk.Entry(root)  # create 5th entry box
canvas1.create_window(270, 200, window=entry6)


def values():
    global ENGINESIZE  # our 1st input variable
    ENGINESIZE = float(entry1.get())

    global CYLINDERS  # our 2nd input variable
    CYLINDERS = float(entry2.get())

    global FUELCONSUMPTION_CITY  # our 3rd input variable
    FUELCONSUMPTION_CITY = float(entry3.get())

    global FUELCONSUMPTION_HWY  # our 4th input variable
    FUELCONSUMPTION_HWY = float(entry4.get())

    global FUELCONSUMPTION_COMB  # our 5th input variable
    FUELCONSUMPTION_COMB = float(entry5.get())

    global FUELCONSUMPTION_COMB_MPG  # our 6th input variable
    FUELCONSUMPTION_COMB_MPG = float(entry6.get())

    Prediction_result = ('Kết quả mô hình hồi quy tuyến tính: ', regr.predict([[ENGINESIZE, CYLINDERS,
                                                                                FUELCONSUMPTION_CITY,
                                                                                FUELCONSUMPTION_HWY,
                                                                                FUELCONSUMPTION_COMB,
                                                                                FUELCONSUMPTION_COMB_MPG]]))
    label_Prediction = tk.Label(root, text=Prediction_result, bg='orange')
    canvas1.create_window(260, 340, window=label_Prediction)


button1 = tk.Button(root, text='Lượng tiêu thụ CO2 ', command=values,
                    bg='orange')  # button to call the 'values' command above
canvas1.create_window(270, 300, window=button1)
root.mainloop()

