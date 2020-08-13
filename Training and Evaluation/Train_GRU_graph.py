import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,GRU
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy,mse
from keras.metrics import accuracy
from sklearn.metrics import f1_score

class Trainer_GRU_sigmoid:

    '''
    sigmoid 함수에서 출력된 값을 binary인 label과 binary crossentropy로 비교하여 학습.
    predict에서는 출력된 sigmoid 값을 threshold value로 분류한 후,
    binary label인 y_test와 비교하여 accuracy 계산.
    '''

    def normalize_data(self,df):
        scaler = MinMaxScaler()
        for i, column in enumerate(df.columns):
            df[df.columns[i]] = scaler.fit_transform(df[df.columns[i]].values.reshape(-1,1))
        return df

    def generator(self,data, lookback=10, delay=5, min_index=0, max_index=None):
        data = data.values #DataFrame을 넘파이로 바꾸기
        max_index = len(data) - delay -1 #4439
        i = min_index + lookback #index10부터 
        samples = np.zeros((len(range(max_index-i+1)), lookback, data.shape[-1])) #(4430, 10, 33)
        targets = np.zeros((len(range(max_index-i+1)),)) #(4430,)
        while i <= max_index:
            indices = range(i-lookback, i)
            samples[i-lookback] = data[indices]
            if data[i+delay][0] - data[i][0] >= 0:
                targets[i-lookback] = 1
            else:
                targets[i-lookback] = 0
            i+=1
        return samples, targets

    def split_data(self, sample, target, train_size_percentage=90, shuffle=True):
        if shuffle is True:
            index = np.arange(sample.shape[0])
            np.random.shuffle(index)
            sample = sample[index]
            target = target[index]
        train_size = train_size_percentage*sample.shape[0]//100
        test_size = sample.shape[0] - train_size
        
        x_train = sample[:train_size]
        y_train = target[:train_size]
        x_test = sample[train_size:]
        y_test = target[train_size:]
        
        return x_train, y_train, x_test, y_test  

    def eval_acc(self, models, x_test_data, y_test_data, threshold=0.5):
        scores = models.predict(x_test_data)
        classified_scores=[]
        for score in scores:
            if score >= threshold:
                classified_scores.append(1)
            else:
                classified_scores.append(0)
        f1 = f1_score(y_test_data, classified_scores)
        acc=0
        for i, classified_score in enumerate(classified_scores):
            if classified_score== y_test_data[i]:
                acc+=1
        accuracy_score = acc/len(y_test_data)
        
        return print('acc: %.3f, f1: %.3f' % (accuracy_score,f1))

class Trainer_GRU_regression:

    '''
    한 개로 출력된 값(회귀)을 변동률과 mse로 비교하여 학습.
    predict에서는 측정된 변동률을 0,1로 분류한 후,
    binary label인 y_test와 비교하여 accuracy 계산.
    '''
    
    def normalize_data(self,df):
        scaler = MinMaxScaler()
        for i, column in enumerate(df.columns[1:]):
            df[df.columns[i+1]] = scaler.fit_transform(df[df.columns[i+1]].values.reshape(-1,1))
        return df

    def generator_change_rate(self,data, lookback=10, delay=5, min_index=0, max_index=None):
        data1 = data.values #DataFrame을 넘파이로 바꾸기
        max_index = len(data1) - delay -1 #4439
        i = min_index + lookback #index10부터 
        samples = np.zeros((len(range(max_index-i+1)), lookback, data1.shape[-1]-1)) #(4430, 10, 33)
        targets = np.zeros((len(range(max_index-i+1)),)) #(4430,)
        while i <= max_index:
            indices = range(i-lookback, i)
            samples[i-lookback] = data1[indices,1:]
            if data.loc[i+delay,'지수종가'] - data.loc[i,'지수종가'] >= 0:
                targets[i-lookback] = 1
            else:
                targets[i-lookback] = 0
            i+=1
        return samples, targets

    def split_data(self,sample, target, train_size_percentage=90, shuffle=True):
        if shuffle is True:
            index = np.arange(sample.shape[0])
            np.random.shuffle(index)
            sample = sample[index]
            target = target[index]
        train_size = train_size_percentage*sample.shape[0]//100
        
        x_train = sample[:train_size]
        y_train = target[:train_size]
        x_test = sample[train_size:]
        y_test = target[train_size:]
        
        return x_train, y_train, x_test, y_test 
    
    def eval_acc(self,models, x_test_data, y_test_data,history,epoch_n):
        scores = models.predict(x_test_data)
        classified_scores=[]
        for score in scores:
            if score >= 0:
                classified_scores.append(1)
            else:
                classified_scores.append(0)
        f1 = f1_score(y_test_data, classified_scores)
        acc=0
        for i, classified_score in enumerate(classified_scores):
            if classified_score== y_test_data[i]:
                acc+=1
        accuracy_score = acc/len(y_test_data)

        y_vloss = history.history['val_loss']
        y_loss = history.history['loss']

        plt.plot(range(epoch_n), y_vloss, marker='.', c='red', label="Validation-set Loss")
        plt.plot(range(epoch_n), y_loss, marker='.', c='blue', label="Train-set Loss")

        plt.legend(loc='best')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        
        return print('acc: %.3f, f1: %.3f' % (accuracy_score,f1))