import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,GRU
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy,mse
from keras.metrics import accuracy
from Train_GRU import Trainer_GRU_regression

train= Trainer_GRU_regression()
data_df = pd.read_csv('C:\\Users\\hit42\\Desktop\\2020 CUAI\\공모전\\preprocessed_data_final.csv')
data_df = data_df.drop(['날짜','연중최고가종목수','연중최저가종목수'], axis=1, inplace=False)

cols= data_df.columns.tolist()
a= cols[:13]
b= cols[14:]
c=[]
c.append(cols[13])
cols =c+a+b
data_df = data_df[cols]

normalized_data_df = train.normalize_data(data_df)
samples, targets = train.generator_change_rate(normalized_data_df,lookback=10, delay=5, min_index=0)
x_train, y_train, x_test, y_test = train.split_data(samples, targets)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

d=0.5
model = Sequential()
model.add(GRU(32,input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(d))
model.add(GRU(64,return_sequences=True))
model.add(Dropout(d))
model.add(GRU(64))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,validation_split=0.2)
train.eval_acc(model, x_test, y_test)
