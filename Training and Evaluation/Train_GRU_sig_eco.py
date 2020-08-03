import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,GRU
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy
from keras.metrics import accuracy
from Train_GRU import Trainer_GRU_sigmoid
from sklearn.metrics import f1_score

train= Trainer_GRU_sigmoid()
data_df = pd.read_csv('C:\\Users\\hit42\\Desktop\\2020 CUAI\\공모전\\added_economics_fillna.csv')
data_df = data_df.drop(['날짜','변동률'], axis=1, inplace=False)
normalized_data_df = train.normalize_data(data_df)
samples, targets = train.generator(normalized_data_df,min_index=0, max_index=3500)
x_train, y_train, x_test, y_test = train.split_data(samples, targets)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

d= 0.5
model = Sequential()
model.add(GRU(32,input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(Dropout(d))
model.add(GRU(64))
model.add(Dropout(d))
model.add(Dense(32, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')

model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,validation_split=0.2)

accuracy = train.eval_acc(model, x_test, y_test)
print(accuracy)