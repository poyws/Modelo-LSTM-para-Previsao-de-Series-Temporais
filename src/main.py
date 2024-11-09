import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

data = pd.read_csv('dados.csv')
data['Data'] = pd.to_datetime(data['Data'])
data.set_index('Data', inplace=True)
data = data[['Valor']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_data_len = int(np.ceil(len(scaled_data) * .8))
train_data = scaled_data[0:int(train_data_len), :]

x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=10)

test_data = scaled_data[train_data_len - 60:, :]
x_test, y_test = [], data['Valor'][train_data_len:].values
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:train_data_len]
valid = data[train_data_len:]
valid.loc[:, 'Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Modelo de Previsão')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.plot(train['Valor'])
plt.plot(valid[['Valor', 'Predictions']])
plt.legend(['Treinamento', 'Validação', 'Previsões'], loc='lower right')
plt.show()
