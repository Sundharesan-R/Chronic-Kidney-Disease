# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 00:00:44 2020

@author: Sundharesan
"""

#Import Libraries
import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import keras as k
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

#load the data 
from google.colab import files 
uploaded = files.upload()      
df = pd.read_csv("kidney_disease.csv")
    
#Print the first 5 rows
df.head()
    
#Get the shape of the data (the number of rows & columns)
df.shape

#Create a list of columns to retain
columns_to_retain = ["sg", "al", "sc", "hemo", "pcv", "wbcc", "rbcc", "htn", "classification"]

#Drop the columns that are not in columns_to_retain
df = df.drop([col for col in df.columns if not col in columns_to_retain], axis=1)
    
# Drop the rows with na or missing values
df = df.dropna(axis=0)

#Transform non-numeric columns into numerical columns
for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])
        
df.head()

#Split the data
X = df.drop(["classification"], axis=1)
y = df["classification"]

#Feature Scaling
x_scaler = MinMaxScaler()
x_scaler.fit(X)
column_names = X.columns
X[column_names] = x_scaler.transform(X)

#Split the data into 80% training and 20% testing 
X_train,  X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.2, shuffle=True)

len(X.columns)

#Build The model
model = Sequential()
model.add(Dense(256, input_dim=len(X.columns), kernel_initializer=k.initializers.random_normal(seed=13), activation="relu"))
model.add(Dense(1, activation="hard_sigmoid"))

#Compile the model
model.compile(loss='binary_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

#Train the model
history = model.fit(X_train, y_train, 
                    epochs=2000, 
                    batch_size=X_train.shape[0]) 

#Save the model
model.save("ckd.model")

#Visualize the model's accuracy and loss
plt.plot(history.history["acc"])
plt.plot(history.history["loss"])
plt.title("model accuracy & loss")
plt.ylabel("accuracy and loss")
plt.xlabel("epoch")
plt.legend(['acc', 'loss'], loc='lower right')
plt.show()

print("---------------------------------------------------------")
print("Shape of training data: ", X_train.shape)
print("Shape of test data    : ", X_test.shape )
print("---------------------------------------------------------")

for model_file in glob.glob("*.model"):
print("Model file: ", model_file)
model = load_model(model_file)
pred = model.predict(X_test)
pred = [1 if y>=0.5 else 0 for y in pred] #Threshold, transforming probabilities to either 0 or 1 depending if the probability is below or above 0.5
scores = model.evaluate(X_test, y_test)
print()
print("Original  : {0}".format(", ".join([str(x) for x in y_test])))
print()
print("Predicted : {0}".format(", ".join([str(x) for x in pred])))
print() 
print("Scores    : loss = ", scores[0], " acc = ", scores[1])
print("---------------------------------------------------------")
print()

        