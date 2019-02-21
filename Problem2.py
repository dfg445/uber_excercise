#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:59:11 2019

@author: yichuanniu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AR
from pandas import Timestamp
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

working_dir = "/Users/yichuanniu/Downloads/uber_exercise_yichuan_niu"


def auto_regressive(data, p = 6):
    """ Auto regressive mode to predict accident rate in Jan 2017 """
    model = AR(data).fit(maxlag = p)
    return model.predict(len(data), len(data), dynamic = False)

def filter_raw_data(data):
    """ Data cleaning and reformatting"""
    for index in list(raw_accident_data.index):
        raw_accident_data.loc[index, "Miles"] = raw_accident_data.loc[index, "Miles"].replace(",", "") 
    raw_accident_data["Miles"] = pd.to_numeric(raw_accident_data["Miles"] )
    raw_accident_data["Month_Ending"] = pd.to_datetime(raw_accident_data["Month_Ending"])

    clean_accident_data = raw_accident_data[raw_accident_data["Miles"] > 0]
    clean_accident_data = clean_accident_data[clean_accident_data["Reported_Accidents"] >= 0]
    clean_accident_data.dropna(inplace = True)
    for index in list(clean_accident_data.index):
        clean_accident_data.loc[index, "Product"] = int(clean_accident_data.loc[index, "Product"] [8:])
        clean_accident_data.loc[index, "City"] = int(clean_accident_data.loc[index, "City"] [5:])
    
    # remove outliers, however, must double inspect the outliters
    clean_accident_data = clean_accident_data[ clean_accident_data["Reported_Accidents"] / clean_accident_data["Miles"] * 1000000 < 2000]
    return clean_accident_data

if __name__ == '__main__':

    # Read in data
    raw_accident_data = pd.read_csv(working_dir + "/mock_accident_data.csv")
    
    # Data cleaning
    clean_accident_data = filter_raw_data(raw_accident_data)
    
    # generate data from plotting
    data_group_by_date = clean_accident_data.groupby(by = ["Month_Ending"]).sum()
    data_group_by_date["Accident_Rate"] = data_group_by_date["Reported_Accidents"] / data_group_by_date["Miles"] * 1000000

    
    data_group_by_segment = clean_accident_data.groupby(by = ["Segment"]).sum()
    data_group_by_segment["Accident_Rate"] = data_group_by_segment["Reported_Accidents"] / data_group_by_segment["Miles"] * 1000000

    data_group_by_city = clean_accident_data.groupby(by = ["City"]).sum()
    data_group_by_city["Accident_Rate"] = data_group_by_city["Reported_Accidents"] / data_group_by_city["Miles"] * 1000000
   
    data_group_by_product = clean_accident_data.groupby(by = ["Product"]).sum()
    data_group_by_product["Accident_Rate"] = data_group_by_product["Reported_Accidents"] / data_group_by_product["Miles"] * 1000000


    predicted_Jan_2017 = auto_regressive(list(data_group_by_date["Reported_Accidents"] / data_group_by_date["Miles"] * 1000000))
    print("Predicted accident rate for Jan. 2017 from autoregressive model:", predicted_Jan_2017)

    
    plt.close('all')

    plt.figure()
    plt.xlabel("Month_Ending")
    plt.ylabel("Accidents per Million Miles")
    plt.title('Overall Accident Rate per Month')
    plt.plot(data_group_by_date.index, data_group_by_date["Accident_Rate"] , 
             "b", linewidth = 3)
    plt.show()
    
    plt.figure()
    plt.xlabel("Segment")
    plt.ylabel("Accidents per Million Miles")
    plt.title('Accident Rate per Segment')
    plt.bar(list(data_group_by_segment.index),list(data_group_by_segment["Accident_Rate"]))
    plt.show()
    
    plt.figure()
    plt.xlabel("City")
    plt.ylabel("Accidents per Million Miles")
    plt.title('Accident Rate per City')
    plt.xticks([index for index in range(1, 77, 2)])
    plt.bar(list(data_group_by_city.index), list(data_group_by_city["Accident_Rate"]))
    plt.show()
    
    plt.figure()
    plt.xlabel("Product")
    plt.ylabel("Accidents per Million Miles")
    plt.title('Accident Rate per Product')
    plt.xticks([index for index in range(1, 23)])
    plt.bar(list(data_group_by_product.index), list(data_group_by_product["Accident_Rate"]))
    plt.show()
    
    plt.figure()
    box_plot = clean_accident_data.copy()
    box_plot["Accident_Rate"] = box_plot["Reported_Accidents"] / box_plot["Miles"] * 1000000
    sns.boxplot(x="Product", y="Accident_Rate", data=box_plot)

    plt.figure()
    plt.xlabel("Month_Ending")
    plt.ylabel("Accidents per Million Miles")
    plt.title('Predicted Accident Rate for Jan 2017')
    plt.plot(data_group_by_date.index,  
    data_group_by_date["Reported_Accidents"] / data_group_by_date["Miles"] * 1000000, 
    "b", linewidth = 3) 
    plt.plot([list(data_group_by_date.index)[-1], Timestamp("2017-01-31")], 
    [list(data_group_by_date["Reported_Accidents"] / data_group_by_date["Miles"] * 1000000)[-1], predicted_Jan_2017], 
    "r", linewidth = 5)
    plt.plot([Timestamp("2017-01-31")],[predicted_Jan_2017], "ro",markersize=10)
    plt.show()
    
    # Creating features and labels
    X, Y = [], []
    for index in clean_accident_data.index:
        X.append([
                clean_accident_data.loc[index, "Segment"][8:], 
               clean_accident_data.loc[index, "City"],
               clean_accident_data.loc[index, "Product"]
               ])
        Y.append(clean_accident_data.loc[index, "Reported_Accidents"] / clean_accident_data.loc[index, "Miles"] * 1000000)
            
    # One-hot encoding the categorical data
    enc = OneHotEncoder(handle_unknown='ignore')
    X_enc = enc.fit_transform(X).toarray()
    
    # Shuffle and split into training and testing sets
    X_shuffle, Y_shuffle = shuffle(X_enc, Y)
    data_spit = int(len(X_shuffle) * 0.85)
    
    X_train = np.array(X_shuffle[:data_spit])
    Y_train = np.array(Y_shuffle[:data_spit])
    
    X_test =  np.array(X_shuffle[data_spit:])
    Y_test = np.array(Y_shuffle[data_spit:])
    
    #Training and predicting
    reg = linear_model.RidgeCV(cv=3, alphas=[0.1])
    reg.fit(X_train, Y_train)
    
    print("R squre metric for testing data:", reg.score(X_test, Y_test))

    fig, ax = plt.subplots()
    plt.xlabel("Index")
    plt.ylabel("Accidents per Million Miles")
    plt.title('Prediction from Ridge Regression')
    
    y_test = reg.predict(X_test)
    ax.plot([index for index in range(len(y_test))], y_test, 'r', 
             label='Predicted Value', linewidth = 3)
    ax.plot([index for index in range(len(y_test))], Y_test, 'b', 
             label='Actual Value', linewidth = 3)
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.show()
