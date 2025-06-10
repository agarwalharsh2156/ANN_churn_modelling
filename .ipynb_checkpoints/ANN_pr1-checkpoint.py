#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

df = pd.read_csv(r"Churn_Modelling - Copy.csv")
df.head()
df.info()
df.tail()
df.shape

#-----------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
LE1 = LabelEncoder()
df['Surname']=LE1.fit_transform(df['Surname'])
df['Gender'].replace({'Female':0, 'Male':1}, inplace=True)
df['Geography'].unique()
df['Geography'].replace({'France':0, 'Spain': 1, 'Germany': 2}, inplace=True)

df
df.shape

X = df.iloc[:,0:-1].values
X

Y = df.iloc[:,-1].values
Y
#-----------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train
X_test
len(X_train)
len(X_test)
#-----------------------------------------------------------

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

df
X_train = sc.fit_transform(X_train)
X_train
X_test = sc.fit_transform(X_test)
X_test
ann= tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
ann.fit(X_train, Y_train, batch_size=10, epochs=100)

Y_pred=ann.predict(X_test)
Y_pred=(Y_pred>0.5)

#-----------------------------------------------------------

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test,Y_pred)